"""Linn/OpenHome Media Player implementation."""

from __future__ import annotations

# mypy: disable-error-code="attr-defined,union-attr"
import asyncio
import functools
import time
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Concatenate
from urllib.parse import urlparse
from uuid import UUID
from xml.etree.ElementTree import Element, ParseError

import defusedxml.ElementTree as DefusedET
from async_upnp_client.exceptions import (
    UpnpActionResponseError,
    UpnpError,
    UpnpResponseError,
)
from async_upnp_client.profiles.ohmedia import (
    InfoState,
    OhmDevice,
    Playlist,
    PlaylistState,
    ProductSourceType,
    ProductState,
    Radio,
    RadioState,
    Service,
    ServiceId,
    TimeState,
    Transport,
    TransportState,
    TransportStateAllowedValues,
    VolumeState,  # or just copy it to helpers if linting protests too much
)
from didl_lite import didl_lite
from music_assistant_models.enums import (
    IdentifierType,
    PlaybackState,
    PlayerFeature,
    PlayerType,
)
from music_assistant_models.errors import InvalidCommand, PlayerUnavailableError
from music_assistant_models.player import PlayerSource

from music_assistant.constants import VERBOSE_LOG_LEVEL
from music_assistant.helpers.upnp import create_didl_metadata
from music_assistant.models.player import DeviceInfo, Player
from music_assistant.providers.openhome_media.constants import (
    CONF_SELECT_DEVICE_SOURCE,
    CONF_SELECT_DEVICE_SOURCE_KEY,
    PLAYER_CONFIG_ENTRIES,
    PLAYLIST,
    RADIO,
    radio_option,
)
from music_assistant.providers.openhome_media.helpers import (
    get_source_index_of_type,
    time_hms_to_sec,
)

if TYPE_CHECKING:
    from async_upnp_client.client import UpnpDevice, UpnpService, UpnpStateVariable
    from music_assistant_models.config_entries import ConfigEntry
    from music_assistant_models.player import PlayerMedia

    from music_assistant.providers.openhome_media.provider import OpenHomePlayerProvider


def catch_request_errors[OpenHomePlayerT: "OpenHomePlayer", **P, R](
    func: Callable[Concatenate[OpenHomePlayerT, P], Awaitable[R]],
) -> Callable[Concatenate[OpenHomePlayerT, P], Coroutine[Any, Any, R | None]]:
    """Catch UpnpError errors."""

    @functools.wraps(func)
    async def wrapper(
        self: OpenHomePlayerT, *args: P.args, **kwargs: P.kwargs
    ) -> R | None:
        """Catch UpnpError errors and check availability before and after request."""
        self.last_command = time.time()
        if self.logger.isEnabledFor(VERBOSE_LOG_LEVEL):
            self.logger.debug(
                "Handling command %s for player %s",
                func.__name__,
                self.display_name,
            )
        if not self.available and func.__name__ not in ("pause", "stop"):
            self.logger.warning(
                "Device disappeared when trying to call %s", func.__name__
            )
            return None
        try:
            return await func(self, *args, **kwargs)
        except UpnpError as err:
            self.force_poll = True
            if self.logger.isEnabledFor(VERBOSE_LOG_LEVEL):
                self.logger.exception("Error during call %s", func.__name__)
            else:
                self.logger.error("Error during call %s: %r", func.__name__, str(err))
        return None

    return wrapper  # type: ignore[return-value]


class OpenHomePlayer(Player):
    """Linn/OpenHome Media Player in Music Assistant."""

    _attr_type = PlayerType.PROTOCOL

    def __init__(
        self,
        provider: OpenHomePlayerProvider,
        player_id: str,
        description_url: str,
        device: OhmDevice | None = None,
    ) -> None:
        """Initialize the Player."""
        super().__init__(provider, player_id)
        self.logger = self.provider.logger.getChild(self.player_id)
        self.profile: OhmDevice | None = device
        self.upnp_device: UpnpDevice | None = None
        self.description_url: str = description_url
        self.last_seen: float = time.time()
        self.lock = asyncio.Lock()  # Held when connecting or disconnecting the device
        self.force_poll = False
        self.state_update_pending: bool = False
        self.state_update_period_ms: int = 1000
        self.product_source_xml: Element | None = (
            None  # state var converted from string
        )
        self._observed_playback_state: PlaybackState | None = None
        self._playing_since: float | None = None
        self._attr_name: str = "Linn/OpenHome Media Player"
        self.player: Player | None = None
        self.playlist_id = None
        self._attr_active_source: str | None = None
        self._device_has_next_media: bool = False
        self.controlled_by_mass: bool = (
            False  # must opt in when mass issues a control command
        )
        self.trailing_id: int = -1  # negative is sentinel for not assigned
        self.selected_play_source: str = RADIO
        # self._attr_source_list = []

    def set_available(self, available: bool) -> None:
        """Set the availability of the player."""
        self._attr_available = available

    @property
    def requires_flow_mode(self) -> bool:
        """Return if the player requires flow mode."""
        return False

    @property
    def has_play_media(self) -> bool:
        """
        Look for usable Play action.

        play_media must select corresponding Play action
        :return: True if Play action found, otherwise False.
        """
        if self.profile.has_transport_play:
            return True
        if self.profile.device.has_service(Service.PLAYLIST):
            svc = self.profile.service(Service.PLAYLIST)
            if svc.has_action(Playlist.PLAY):
                return True
        if self.profile.device.has_service(Service.RADIO):
            svc = self.profile.service(Service.RADIO)
            if svc.has_action(Radio.PLAY):
                return True
        return False

    async def setup(self) -> bool:
        """
        Set up player in MA.

        :return: True if setup was successful, False if device should be ignored.
        """
        self.logger.debug("SETUP: %s", self.player_id)

        await self._device_connect()

        if self.profile and not self.has_play_media:
            self.logger.debug("Ignoring %s - no play capability", self.profile.name)
            return False

        self._attr_name = self.profile.name
        self.set_static_attributes()
        await self.mass.players.register_or_update(self)

        assert self.mass.players is not None
        self._attr_active_source = self.mass.players.get_player(
            self.player_id
        ).protocol_parent_id

        self.selected_play_source = str(
            await self.mass.config.get_player_config_value(
                player_id=self.player_id, key=CONF_SELECT_DEVICE_SOURCE_KEY
            )
        )

        await self.profile.async_update_state_variables()
        await self.set_dynamic_attributes()
        return True

    def set_static_attributes(self) -> None:
        """Set static attributes."""
        self._attr_poll_interval = self.poll_interval
        self._set_player_features()

    async def set_dynamic_attributes(self) -> None:
        """
        Update/set MA attributes from state variables.

        Use after poll all variables has occurred
        When subscribed each update should be handled by service event handler
        """
        self.logger.warning("SET_DYNAMIC_ATTRIBUTES STARTED")
        # updates all of the following
        # _attr_available
        # _attr_name
        # _attr_volume_level
        # _attr_volume_muted
        # _attr_playback_state
        # _attr_current_media
        # _attr_source_list
        # _attr_active_source
        # _attr_elapsed_time
        # _attr_elapsed_time_last_updated

        available = self.profile is not None and self.profile.device.available
        self._attr_available = available
        if not available:
            self.logger.debug("set_dynamic_attributes NOT AVAILABLE - returning")
            return
        assert self.profile is not None  # for type checking

        if self.force_poll:
            await (
                self.profile.async_update_state_variables()
            )  # poll all state variables

        self._attr_powered = not self.profile.product_standby
        self._attr_volume_muted = self.profile.is_muted

        # should a device report an unknown volume, retain unknown instead of collapsing to 0/unmuted
        volume_level = self.profile.volume
        self._attr_volume_level = (
            int(volume_level) if volume_level is not None else None
        )
        # self.logger.warning("SET_DYNAMIC_ATTRIBUTES 1a")

        # determine playback state
        _playback_state = self._transport_state_to_playback_state(
            self.profile.transport_state
        )

        if _playback_state is not None:
            prev_playback_state = self._observed_playback_state
            self._observed_playback_state = _playback_state

            if _playback_state != PlaybackState.PLAYING:
                self._playing_since = None
            elif prev_playback_state not in (None, PlaybackState.PLAYING):
                self._playing_since = time.time()
            self._attr_playback_state = _playback_state

        # current media
        media_title = media_artist = media_album = media_image_url = media_duration = (
            None
        )
        # self.logger.warning("SET_DYNAMIC_ATTRIBUTES 2a %s", self.profile.info_metadata)

        if (
            self.profile.info_metadata is not None
            and len(self.profile.info_metadata) > 0
        ):
            try:
                metadata = didl_lite.from_xml_string(
                    self.profile.info_metadata, strict=False
                )  # Linn metadata
                # self.logger.warning("SET_DYNAMIC_ATTRIBUTES 2b")
            except ParseError as err:
                # log the error, player will be updated with media_* values equal to None
                self.logger.debug(
                    "Ignoring malformed media metadata from device %s: %s: %s",
                    self.display_name,
                    err,
                    self.profile.info_metadata,
                )
            else:
                # self.logger.warning("SET_DYNAMIC_ATTRIBUTES 2c")
                if metadata is not None and len(metadata) > 0:
                    media_title = metadata[0].title  # type: ignore[possibly-undefined]
                    media_artist = metadata[0].artist  # type: ignore[possibly-undefined]
                    media_album = metadata[0].album
                    media_image_url = metadata[0].albumArtURI
                    res = metadata[0].res[0]
                    media_duration = res.duration
                    # self.logger.warning("SET_DYNAMIC_ATTRIBUTES 2d %s", media_title)

        # self.logger.warning("SET_DYNAMIC_ATTRIBUTES 2e info_uri: %s", self.profile.info_uri)
        _track_uri: str = (
            str(self.profile.info_uri) if self.profile.info_uri is not None else ""
        )
        # self.logger.warning("SET_DYNAMIC_ATTRIBUTES 3b media_duration %s", media_duration)

        duration = (
            time_hms_to_sec(media_duration) if media_duration is not None else None
        )
        # self.logger.warning("SET_DYNAMIC_ATTRIBUTES 3b duration_s %s", duration_s)

        self.set_current_media(
            clear_all=True,
            uri=_track_uri,
            title=media_title,
            artist=media_artist,
            album=media_album,
            image_url=media_image_url,
            duration=duration,
        )
        # self.logger.warning("SET_DYNAMIC_ATTRIBUTES 4 %s", self._attr_current_media)

        if self.product_source_xml:
            self._attr_source_list = self._source_list_from_source_xml(
                self.product_source_xml
            )
        else:
            self._attr_source_list = []
        self.logger.warning("SOURCES %s", self._attr_source_list[0].name)
        self.logger.warning("SET_DYNAMIC_ATTRIBUTES OK")

    @property
    def poll_interval(self) -> int:
        """Return the interval in seconds to poll the player for state updates."""
        return 5 if self._attr_playback_state == PlaybackState.PLAYING else 30

    async def get_config_entries(self) -> list[ConfigEntry]:
        """Return all (provider/player specific) Config Entries for the given player (if any)."""
        config_entries: list[ConfigEntry] = [*PLAYER_CONFIG_ENTRIES]

        if (has_radio := self.profile.has_source_type(ProductSourceType.RADIO)) is None:
            await self.profile._async_poll_state_variables(
                Service.PRODUCT, ProductState.SOURCE_XML
            )
            has_radio = self.profile.has_source_type(ProductSourceType.RADIO)

        if has_radio:
            CONF_SELECT_DEVICE_SOURCE.default_value = RADIO
        else:  # expected that most devices do have radio so opt out rather than opt in
            CONF_SELECT_DEVICE_SOURCE.default_value = PLAYLIST
            radio_option.disabled = True
            radio_option.disabled_reason = "Player does not support Radio"

        return config_entries

    # region COMMANDS
    @catch_request_errors
    async def power(self, powered: bool) -> None:
        """Handle POWER command on the player."""
        self.logger.debug("Command POWER %s for player %s", powered, self.display_name)
        await self.profile.async_product_set_standby(not powered)

    @catch_request_errors
    async def volume_set(self, volume_level: int) -> None:
        """Handle VOLUME SET command on the player."""
        self.logger.debug(
            "Command VOLUME_SET level %s for player %s",
            volume_level,
            self.display_name,
        )

        await self.profile.async_volume_set(volume_level)
        self.update_state()

    @catch_request_errors
    async def volume_mute(self, muted: bool) -> None:
        """Handle VOLUME MUTE command on the player."""
        self.logger.debug(
            "Command VOLUME_MUTE %s for player %s",
            muted,
            self.display_name,
        )

        await self.profile.async_volume_set_mute(muted)
        # self.update_state()
        await self._update_player()

    @catch_request_errors
    async def play(self) -> None:
        """Play command."""
        self.logger.debug("Command PLAY for player %s", self.display_name)
        try:
            await self.profile.async_play()
        except UpnpError:
            self.logger.warning(
                "Could not execute PLAY command on player %s", self.display_name
            )

    @catch_request_errors
    async def stop(self) -> None:
        """Stop command."""
        self.logger.debug("Command STOP for player %s", self.display_name)
        if self.active_source_type == RADIO:
            self.cancel_next_media_timer()
        if self.active_source_type == PLAYLIST:
            # TODO tidy up tracks if playlist
            pass
        try:
            await self.profile.async_stop()
        except UpnpError:
            self.logger.debug(
                "Could not execute STOP command on player %s", self.display_name
            )


    @catch_request_errors
    async def pause(self) -> None:
        """Pause command."""
        self.logger.debug("Command PAUSE for player %s", self.display_name)
        # TODO Pause/Play doesn't work
        #  Play adds new tracks
        #  old tracks not removed
        # Service has Pause is different from CanPause

        # Get CAN_PAUSE capability, polling if necessary
        can_pause = self.profile.get_state_variable_value(
            Service.TRANSPORT, TransportState.CAN_PAUSE
        )
        if can_pause is None:
            await self.profile._async_poll_state_variables(
                Service.TRANSPORT, Transport.STREAM_INFO
            )
            can_pause = self.profile.get_state_variable_value(
                Service.TRANSPORT, TransportState.CAN_PAUSE
            )
        # If device supports pause, use pause; otherwise fall back to stop
        if can_pause:
            try:
                await self.profile.async_pause()
            except UpnpError:
                self.logger.warning(
                    "Could not execute PAUSE command on player %s", self.display_name
                )
        else:
            try:
                await self.stop()
            except UpnpError:
                self.logger.warning(
                    "Could not execute STOP command on player %s", self.display_name
                )

        if self.active_source_type == RADIO:
            self.cancel_next_media_timer()
        if self.active_source_type == PLAYLIST:
            # TODO Handle timers etc. for Playlist
            pass

    @catch_request_errors
    async def next_track(self) -> None:
        """Next command."""
        if self.active_source_type == PLAYLIST:
            # TODO tidy up tracks if playlist
            await self.profile.async_playlist_next()

    @catch_request_errors
    async def previous_track(self) -> None:
        """Previous command."""
        if self.active_source_type == PLAYLIST:
            # TODO tidy up tracks if playlist
            await self.profile.async_playlist_previous()

    @catch_request_errors
    async def seek(self, position: int) -> None:
        """SEEK command on the player."""
        if self.profile.has_transport_seek_second_absolute:
            stream_id = self.profile.transport_stream_id
            if stream_id:
                await self.profile.async_transport_seek_second_absolute(
                    stream_id, position
                )
                return
        if self.active_source_type == ProductSourceType.RADIO:
            await self.profile.async_radio_seek_second_absolute(position)
        if self.active_source_type == ProductSourceType.PLAYLIST:
            await self.profile.async_playlist_seek_second_absolute(position)

    @catch_request_errors
    async def play_media(self, media: PlayerMedia) -> None:
        """Play media command."""
        self.logger.debug("Command PLAY_MEDIA for player %s", self.display_name)
        self.controlled_by_mass = True
        # always stop any scheduled enqueue_next_media task
        await self.stop()

        url = await self.provider.mass.streams.resolve_stream_url(self.player_id, media)
        didl_metadata = create_didl_metadata(media, url)
        # self.logger.warning("didl_metadata: %s", didl_metadata)
        # title = media.title or media.uri
        # optimistically set the state here to help in case of a player
        # that is slow or failing to report state changes.
        prev_state = self._attr_playback_state

        self.selected_play_source = str(
            await self.mass.config.get_player_config_value(
                player_id=self.player_id, key=CONF_SELECT_DEVICE_SOURCE_KEY
            )
        )

        try:
            if self.selected_play_source == RADIO:
                # workaround to avoid hung buffering problem with Linn DSM - flip source away from Radio
                if self.profile.product_source_xml is not None:
                    radio_index = get_source_index_of_type(
                        self.profile.product_source_xml, RADIO
                    )
                    if radio_index and radio_index != 0:
                        new_index = 0  # expect this to be Playlist and always available
                await self.profile.async_product_set_source_index(new_index)  # type: ignore[possibly-undefined]
                await self.profile.async_radio_set_channel(url, didl_metadata)
                await asyncio.sleep(
                    0.5
                )  # allow reaction time for device to flip back to Radio
                await self.profile.async_radio_play()
            elif self.selected_play_source == PLAYLIST:  # use Playlist
                # assume control of Playlist
                # TODO tidy up tracks if playlist
                #  and playing from pause
                last_id = await self.profile.async_playlist_last_id()
                new_id = (
                    await self.profile.async_playlist_insert(
                        last_id, url, didl_metadata
                    )
                ).get("NewId")
                if new_id is not None:
                    await self.profile.async_playlist_seek_id(
                        new_id
                    )  # play track at new_id
        except Exception:
            self._attr_playback_state = prev_state  # rollback optimistic state
            raise
        else:
            self.set_current_media(
                uri=url,
                title=media.title,
                artist=media.artist,
                album=media.album,
                image_url=media.image_url,
                duration=media.duration,
                source_id=media.source_id,
                clear_all=True,
            )
            self._attr_playback_state = PlaybackState.PLAYING
            self._attr_elapsed_time = 0
            self._attr_elapsed_time_last_updated = time.time()
        self.update_state(force_update=True)

    @catch_request_errors
    async def enqueue_next_media(self, media: PlayerMedia) -> None:
        """Handle enqueuing of the next queue item on the player."""
        assert self.profile is not None  # for type checking
        next_track_url = await self.provider.mass.streams.resolve_stream_url(
            self.player_id, media
        )
        didl_metadata = create_didl_metadata(media, next_track_url)

        active_queue = self.mass.player_queues.get_active_queue(self.player_id)
        if active_queue is None:
            self.logger.debug("No active queue")
            return
        active_queue_id = active_queue.queue_id
        queue = self.mass.player_queues.get(active_queue_id)

        remaining_time: float = 0.0
        if queue.current_item.duration is not None:
            remaining_time = (
                float(queue.current_item.duration) - queue.corrected_elapsed_time
            )
            remaining_time = max(0, remaining_time)

        self.logger.debug("ENM: active_source_type %s", self.active_source_type)
        if self.active_source_type == RADIO:
            # schedule a play_media for when track has ended
            if active_queue.state == PlaybackState.PLAYING:
                self.mass.call_later(
                    remaining_time,
                    self.play_media,
                    media,
                    task_id=f"ohm_enqueue_{active_queue_id}",
                )
        elif self.active_source_type == PLAYLIST:
            # append track to Playlist
            self.logger.debug("ENM: append track to Playlist %s", self.active_source_type)

            current_id = (await self.profile.async_playlist_id())["Value"]
            self.trailing_id = current_id
            current_uri = (await self.profile.async_playlist_read(current_id)).get(
                "Uri"
            )
            last_id = await self.profile.async_playlist_last_id()
            #     # only enqueue if there's something to play, and we are playing
            #     # and there are no more tracks
            #     # and if next track is not a repeat of current track
            self.logger.debug("ENM: %s %s %s %s", next_track_url,current_uri,self.profile.transport_state,current_id, )
            if (
                next_track_url is not None
                # and self.profile.transport_state == TransportStateAllowedValues.PLAYING
                and current_id == last_id
                and current_uri != next_track_url
            ):  # at end of track playlist
                self.logger.debug("ENM: async_playlist_insert")
                await self.profile.async_playlist_insert(
                    last_id, next_track_url, didl_metadata
                )
                self._device_has_next_media = True
                self.logger.warning(
                    "ENQUEUE NEXT MEDIA: %s - %s", media.artist, media.title
                )

        self.update_state()

    def cancel_next_media_timer(self) -> None:
        """Cancel the next media scheduled task on the player."""
        assert self.profile is not None
        # active_queue = self.mass.player_queues.get_active_queue(self.player_id)
        active_queue = self.mass.player_queues.get_active_queue(self.player_id)
        if active_queue is not None:
            self.mass.cancel_timer(task_id=f"ohm_enqueue_{active_queue.queue_id}")
            # self.mass.cancel_task(task_id=f"ohm_next_track_{active_queue.queue_id}")

    @catch_request_errors
    async def select_source(self, source_name: str) -> None:
        """
        Handle SELECT SOURCE command on the player.

        :param source_name: The name of the source to select, as defined by source_list.
        """
        new_source = next(
            (x for x in self.source_list if x.name.lower() == source_name.lower()), None
        )
        if new_source:
            await self.profile.async_product_set_source_index(int(new_source.id))

    @property
    def source_list(self) -> list[PlayerSource]:
        """Return list of available (native) sources for this player."""
        return self._attr_source_list

    async def poll(self) -> None:
        """Poll player for all state variables (fallback mode only)."""
        self.force_poll = False
        # try to reconnect the device if the connection was lost
        if self.profile is None:
            if not self.force_poll:
                return
            try:
                await self._device_connect()
            except UpnpError as err:
                raise PlayerUnavailableError from err
        elif self.profile.is_subscribed:
            self.force_poll = False
            return

        assert self.profile is not None
        try:
            now = time.time()
            do_ping = (
                self.force_poll or (now - self.last_seen) > 60
            )  # TODO determine best value for this

            with suppress(ValueError, ParseError):
                self.logger.debug("Poll ALL state variables: %s", self._attr_name)
                await self.profile.async_update_state_variables(do_ping=do_ping)
        except UpnpError as err:
            # self.logger.debug("Device unavailable: %r", err)
            await self._device_disconnect()
            raise PlayerUnavailableError from err
        else:
            self.last_seen = now if do_ping else self.last_seen
        finally:
            self.force_poll = False

    async def on_unload(self) -> None:
        """Handle logic when the player is unloaded from the Player controller."""
        await super().on_unload()
        await self._device_disconnect()
        self.logger.debug("Player unloaded: %s", self.name)

    # endregion

    # region Linn/OpenHome Media specific helper functions
    @staticmethod
    def get_mac_from_udn(udn: str) -> str | None:
        """Return a mac-address-like string from the UDN of the device."""
        udn = udn.removeprefix("uuid:")
        mac_like = udn[udn.find("-") + 1 : udn.rfind("-")]
        mac_like = mac_like.replace("-", "")
        # Format string like a MAC address i.e. XX:XX:XX:XX:XX:XX
        mac_like = ":".join(mac_like[i : i + 2].upper() for i in range(0, 12, 2))
        if len(mac_like) == 17:
            return mac_like
        return None

    @staticmethod
    def is_valid_uuid(uuid_string: str | None) -> bool:
        """Check string is a valid UUID."""
        if uuid_string is not None:
            try:
                UUID(uuid_string)
                return True
            except ValueError:
                return False
        return False

    @staticmethod
    def _source_list_from_source_xml(source_xml: Element | None) -> list[PlayerSource]:
        """Convert source XML into MA source list."""
        player_source_list: list[PlayerSource] = []
        if isinstance(source_xml, Element):
            for _index, element in enumerate(source_xml):
                visible: str | None = element.findtext("Visible")
                if visible and visible.lower().strip() in ("true", "1"):
                    source_type = element.findtext("Type", default="External")
                    if source_type in [
                        "Playlist",
                        "Radio",
                        "Analog",
                        "Digital",
                        "Hdmi",
                        "NetAux",
                        "Receiver",
                        "UpnpAv",
                        "External",
                    ]:
                        source_entry = PlayerSource(
                            id=source_type.lower(),  # to be external, value should be in constants EXTERNAL_SOURCES
                            name=element.findtext("Name", default="Unknown"),
                            can_play_pause=bool(source_type in ("Playlist", "Radio")),
                            can_seek=bool(source_type == "Playlist"),
                            can_next_previous=bool(source_type == "Playlist"),
                            passive=False,  # all visible sources so any one can be actively selected
                        )
                        player_source_list.append(source_entry)
        return player_source_list

    @staticmethod
    def _transport_state_to_playback_state(
        transport_state: str | None,
    ) -> PlaybackState | None:
        """
        Return MA playback state from Linn/OpenHome Media device transport state.

        None is used to as a no-op instruction i.e. ignore the state reported
        """
        match transport_state:
            case TransportStateAllowedValues.PLAYING:
                return PlaybackState.PLAYING
            case TransportStateAllowedValues.PAUSED:
                return PlaybackState.PAUSED
            case TransportStateAllowedValues.STOPPED:
                return PlaybackState.IDLE
            case TransportStateAllowedValues.BUFFERING:
                return None  #
            case TransportStateAllowedValues.WAITING:
                return None  #
            case _:
                # not given an allowed value
                return PlaybackState.UNKNOWN

    # endregion

    async def _device_connect(self) -> None:
        """Connect Linn/OpenHome Media Device."""
        # self.logger.warning("_device_connect: %s", self.description_url)

        async with self.lock:
            if self.profile is not None:
                self.logger.debug("Trying to connect when device already connected")
                return

            # Connect to the base UPNP device
            if TYPE_CHECKING:
                assert isinstance(self.provider, OpenHomePlayerProvider)
            # if self.upnp_device is None:
            self.upnp_device = await self.provider.upnp_factory.async_create_device(
                self.description_url
            )

            self._attr_device_info = DeviceInfo(
                model=self.upnp_device.model_name,
                manufacturer=self.upnp_device.manufacturer,
            )

            # Create profile wrapper
            if OhmDevice.is_profile_device(self.upnp_device):
                self.profile = OhmDevice(
                    self.upnp_device, self.provider.notify_server.event_handler
                )
            else:
                return

            # replace default services with just those of interest
            self.profile._SERVICE_TYPES = {
                "Info": {"urn:av-openhome-org:service:Info:1"},
                "Playlist": {"urn:av-openhome-org:service:Playlist:1"},
                "Product": {
                    "urn:av-openhome-org:service:Product:4",
                    "urn:av-openhome-org:service:Product:3",
                    "urn:av-openhome-org:service:Product:2",
                    "urn:av-openhome-org:service:Product:1",
                },
                "Radio": {
                    "urn:av-openhome-org:service:Radio:2",
                    "urn:av-openhome-org:service:Radio:1",
                },
                "Time": {"urn:av-openhome-org:service:Time:1"},
                "Transport": {"urn:av-openhome-org:service:Transport:1"},
                "Volume": {
                    "urn:av-openhome-org:service:Volume:4",
                    "urn:av-openhome-org:service:Volume:3",
                    "urn:av-openhome-org:service:Volume:2",
                    "urn:av-openhome-org:service:Volume:1",
                },
            }

            # Subscribe to event notifications
            try:
                self.profile.on_event = self._handle_event
                # doesn't seem to be possible to alter auto_resubscribe timeout elegantly
                # should be set to 30 at declaration and use device value if provided
                # hoping that at least one of these assignments work
                # async_upnp_client.profiles.profile.SUBSCRIBE_TIMEOUT = timedelta(minutes=30)
                # self.profile.SUBSCRIBE_TIMEOUT = timedelta(minutes=30)  # default currently is 9 but Linn caps at 30
                await self.profile.async_subscribe_services(auto_resubscribe=True)
            except UpnpResponseError:
                # Device rejected subscription request.
                # This is OK, variables will be polled instead.
                self.force_poll = True
            except UpnpError:
                # Don't leave the device half-constructed
                self.profile.on_event = None
                self.profile = None
                raise
            else:
                self.logger.debug(
                    "async_subscribe_services was successful %s", self._attr_device_info
                )
                # Identifiers in descending priority MAC_ADDRESS, UUID, IP_ADDRESS
                # MAC_ADDRESS is extracted from UUID if format of UDN is UUID-like
                # MAC_ADDRESS will be validated by player controller enrich_device_mac_address
                # OpenHome Media Software Player uses machine name so will be excluded
                assert self.profile is not None
                if OpenHomePlayer.is_valid_uuid(self.profile.device.udn):
                    # MAC address frequently part of udn - will be checked by arp later
                    mac_address = OpenHomePlayer.get_mac_from_udn(
                        self.profile.device.udn
                    )
                    self._attr_device_info.add_identifier(
                        IdentifierType.MAC_ADDRESS, mac_address
                    )

                    # Add UDN as identifier type uuid for identifying player across protocols
                    # Strip the common to all "uuid:" prefix if present
                    self._attr_device_info.add_identifier(
                        IdentifierType.UUID,
                        self.profile.device.udn.removeprefix("uuid:"),
                    )

                # Try to extract just the IP from the URL for device matching
                # All currently known examples have a higher priority identifier available
                ip_address = (
                    self.profile.device.presentation_url or self.description_url
                )
                with suppress(ValueError):
                    parsed = urlparse(ip_address)
                    if parsed.hostname:
                        self._attr_device_info.add_identifier(
                            IdentifierType.IP_ADDRESS, parsed.hostname
                        )

    async def _device_disconnect(self) -> None:
        """Destroy connections to the device."""
        async with self.lock:
            if not self.profile:
                self.logger.debug(
                    "Cannot disconnect from a device that's not connected"
                )
                return

            self.logger.debug("Disconnecting from %s", self.profile.name)

            self.profile.on_event = None
            old_device = self.profile
            self.profile = None
            self.set_available(False)
            await old_device.async_unsubscribe_services()
        self.update_state()

    async def _deferred_update(self) -> None:
        """Defer update for a period."""
        await asyncio.sleep(self.state_update_period_ms / 1000.0)
        self.update_state()

    async def _delete_track(self, playlist_id: int):
        try:
            await self.profile.async_playlist_delete_id(playlist_id)
        except UpnpActionResponseError as err:
            self.logger.debug("_advance_queue %s %s", err.error_desc, playlist_id)

    async def _advance_queue(self, queue_id: str) -> None:
        """
        Advance queue for given queue_id.

        :param queue_id: the id of the queue to advance
        """
        self.logger.warning("ADVANCE: CALLED %s", queue_id)

        # extracted from state management part of player queues controller next()
        self.mass.player_queues._check_player_permission(queue_id)
        if (queue := self.mass.player_queues.get(queue_id)) is None or not queue.active:
            raise InvalidCommand(f"Queue {queue_id} is not active")
        self.mass.player_queues._set_transitioning(queue_id, True)
        idx = self.mass.player_queues._queue_data[queue_id].queue.current_index
        if idx is None:
            self.logger.warning("Queue %s has no current index", queue.display_name)
            self.mass.player_queues._set_transitioning(queue_id, False)
            return

        # region REPORT
        # self.logger.warning("ADVANCE: queue: %s %s %s %s", _queue.items, _queue.state, _queue.current_index, _queue.current_item.queue_item_id)
        # self.logger.warning("ADVANCE: _DEVICE_HAS_NEXT_MEDIA %s", self._device_has_next_media)
        # self.logger.warning("ADVANCE: ATTR PLAYBACK STATE: %s", self._attr_playback_state)
        # self.logger.warning("ADVANCE: QUEUE PLAYBACK STATE: %s", _queue.state)
        # self.logger.warning("ADVANCE: BASE URL: %s", self.mass.streams.base_url)
        # self.logger.warning("ADVANCE: DS URI: %s", uri)
        # endregion

        base_url = self.mass.streams.base_url
        uri = self.profile.get_state_variable_value(Service.INFO, InfoState.URI)
        if uri is not None:
            self.controlled_by_mass = uri.startswith(base_url)
            self.logger.warning("ADVANCE: MASS CONTROL IS: %s", self.controlled_by_mass)

        if not self.controlled_by_mass:
            self.logger.warning("ADVANCE: NOT MASS CONTROL, RETURNING")
            return

        # ACTION
        is_last_track = (
            queue.items == (queue.current_index + 1)
            if queue.current_index is not None
            else None
        )
        self.logger.warning("ADVANCE: IS LAST TRACK %s %s", is_last_track, self._attr_playback_state)
        is_idle = self._attr_playback_state == PlaybackState.IDLE

        if (
            self._device_has_next_media and not is_idle and not is_last_track
        ):  # order by most frequently False?
            self.logger.warning("ADVANCE: ADVANCING")
            self._device_has_next_media = (
                False  # reset flag, to be set by enqueue_next_media
            )
            next_index = self.mass.player_queues._get_next_index(queue_id, idx, True)
            queue.current_index = next_index
            queue.current_item = self.mass.player_queues.get_item(queue_id, next_index)
            queue.elapsed_time = 0
            queue.elapsed_time_last_updated = time.time()
            self.mass.player_queues.signal_update(queue_id)
            if queue_player := self.mass.players.get_player(queue_id, True):
                queue_player.update_state()
            self.update_state()
            # remove played mass track from Playlist
            self.mass.call_later(
                delay=3,
                target=self._delete_track,
                playlist_id=self.trailing_id,
                task_id=f"ohm_delete_track_{self.trailing_id}",
            )
            return

        self.logger.warning("ADVANCE: PRE END OF QUEUE %s %s", is_last_track, self._device_has_next_media)
        if is_last_track and not self._device_has_next_media:
            self.logger.warning("ADVANCE: *END OF QUEUE* %s %s", self._attr_playback_state, self._attr_elapsed_time)
            # TODO: why does it sometimes wait for the 5 seconds before stopping
            last_id = await self.profile.async_playlist_last_id()
            self._attr_current_media = None
            self._attr_elapsed_time = None
            self._attr_elapsed_time_last_updated = None

            self._attr_playback_state = PlaybackState.IDLE
            await self.mass.player_queues.stop(queue_id)
            # await self.stop()
            self.mass.player_queues.signal_update(queue_id)
            if queue_player := self.mass.players.get_player(queue_id, True):
                queue_player.update_state()
            self.update_state()

            # remove final played mass track from Playlist
            self.mass.call_later(
                delay=3,
                target=self._delete_track,
                playlist_id=last_id,
                task_id=f"ohm_delete_track_{last_id}",
            )

    async def _handle_info_event(
        self, state_variables: Sequence[UpnpStateVariable[Any]]
    ) -> None:
        """
        Handle state variable changes from Info service

        :param state_variables: the state variables whose values have changed
        """
        # TODO isn't this handled by play media?  Would this just be for Playlist sync
        if self.controlled_by_mass:
            media_title = media_artist = media_album = media_image_url = (
                media_duration
            ) = None
            duration: int | None = None
            track_uri: str = str(self._attr_current_media.uri)
            for sv in state_variables:
                if sv.value is not None:
                    match sv.name:
                        case InfoState.DURATION:
                            duration = sv.value
                        case InfoState.METADATA:
                            try:
                                metadata = didl_lite.from_xml_string(
                                    str(sv.value), strict=False
                                )  # Linn metadata
                            except ParseError as err:
                                # log the error, player will be updated with media_* values equal to None
                                self.logger.debug(
                                    "Ignoring malformed media metadata from device %s: %s: %s",
                                    self.display_name,
                                    err,
                                    sv.value,
                                )
                            else:
                                if metadata is not None and len(metadata) > 0:
                                    media_title = metadata[0].title  # type: ignore[possibly-undefined]
                                    media_artist = metadata[0].artist  # type: ignore[possibly-undefined]
                                    media_album = metadata[0].album
                                    media_image_url = metadata[0].albumArtURI
                                    res = metadata[0].res[0]
                                    media_duration = res.duration
                                    # self.logger.warning("SET_DYNAMIC_ATTRIBUTES 2d %s", media_title)
                        case InfoState.URI:
                            # URI has changed
                            track_uri: str = str(sv.value)
                        case _:
                            pass
            # TODO decide which duration to use
            if media_duration is None:
                duration_s = duration
            else:
                duration_s = (
                    time_hms_to_sec(media_duration)
                    if media_duration is not None
                    else None
                )

            if track_uri is not None:
                self.set_current_media(
                    clear_all=True,
                    uri=track_uri,
                    title=media_title,
                    artist=media_artist,
                    album=media_album,
                    image_url=media_image_url,
                    duration=duration_s,
                )

    async def _handle_playlist_event(
        self, state_variables: Sequence[UpnpStateVariable[Any]]
    ) -> None:
        """
        Handle state variable changes from Playlist service

        :param state_variables: the state variables whose values have changed
        """
        if self.controlled_by_mass:
            for sv in state_variables:
                match sv.name:
                    case PlaylistState.TRANSPORT_STATE:
                        self.logger.warning(
                            "PlaylistState.TRANSPORT_STATE: %s %s",
                            sv.value,
                            self.controlled_by_mass,
                        )
                        _playback_state = self._transport_state_to_playback_state(sv.value)
                        if _playback_state is not None:
                            self._attr_playback_state = _playback_state
                        prev_playback_state = self._observed_playback_state
                        self._observed_playback_state = self._attr_playback_state
                        if self._attr_playback_state != PlaybackState.PLAYING:
                            self._playing_since = None
                        elif prev_playback_state not in (None, PlaybackState.PLAYING):
                            self._playing_since = time.time()
                    case PlaylistState.REPEAT:
                        self._attr_repeat_state = sv.value
                    case PlaylistState.SHUFFLE:
                        self._attr_shuffle_state = sv.value
                    case PlaylistState.ID:
                        self.playlist_id = sv.value
                    case PlaylistState.ID_ARRAY:
                        pass
                    case _:
                        self.logger.debug(
                            "Unhandled Playlist State Variable %s", sv.name
                        )

    async def _handle_product_event(
        self, state_variables: Sequence[UpnpStateVariable[Any]]
    ) -> None:
        """
        Handle state variable changes from Product service

        :param state_variables: the state variables whose values have changed
        """
        for sv in state_variables:
            if sv.value is not None:
                match sv.name:
                    case ProductState.STANDBY:
                        self._attr_powered = not sv.value
                    case ProductState.SOURCE_INDEX:
                        source = await self.profile.async_product_source(sv.value)
                        self.active_source_type = source.get("Type")
                    case ProductState.SOURCE_XML:
                        # this should be fairly static
                        try:
                            if sv.value:
                                self.product_source_xml = DefusedET.fromstring(
                                    str(sv.value)
                                )
                        except (
                            ParseError,
                            AttributeError,
                            IndexError,
                            KeyError,
                            TypeError,
                        ):
                            self.logger.debug("Unable to process Source XML %s", sv.value)
                        else:
                            self._attr_source_list = self._source_list_from_source_xml(
                                self.product_source_xml
                            )
                    case _:
                        pass

    async def _handle_radio_event(
        self, state_variables: Sequence[UpnpStateVariable[Any]]
    ) -> None:
        """
        Handle state variable changes from Radio service

        :param state_variables: the state variables whose values have changed
        """
        if self.controlled_by_mass:
            for sv in state_variables:
                match sv.name:
                    case RadioState.TRANSPORT_STATE:
                        _playback_state = self._transport_state_to_playback_state(sv.value)
                        if _playback_state is not None:
                            self._attr_playback_state = _playback_state
                    case _:
                        pass

    async def _handle_time_event(
        self, state_variables: Sequence[UpnpStateVariable[Any]]
    ) -> None:
        """
        Handle state variable changes from Time service

        :param state_variables: the state variables whose values have changed
        """
        active_queue = self.mass.player_queues.get_active_queue(self.player_id)
        if self.controlled_by_mass:
            sv_dict = {}
            for sv in state_variables:
                match sv.name:
                    case TimeState.TRACK_COUNT:
                        sv_dict.update({sv.name: sv.value})
                    case TimeState.DURATION:
                        if self._attr_current_media:
                            self._attr_current_media.duration = sv.value
                    case TimeState.SECONDS:
                        sv_dict.update({sv.name: sv.value})
                        self._attr_elapsed_time = sv.value
                        self._attr_elapsed_time_last_updated = time.time()
                        self.update_state()
                        # only log significant differences in times - remove before release
                        # TODO reconcile time differences
                        if (
                            self._attr_elapsed_time is not None
                            and active_queue
                            and abs(
                                self._attr_elapsed_time
                                - active_queue.corrected_elapsed_time
                            )
                            > 1.0
                        ):
                            self.logger.debug(
                                "TIME: %s %s",
                                self._attr_elapsed_time,
                                active_queue.corrected_elapsed_time,
                            )
                    case _:
                        pass
            # use the following condition as a gate for end of track
            if TimeState.TRACK_COUNT in sv_dict and sv_dict.get(TimeState.SECONDS) == 0:
                self.logger.warning("ADVANCE QUEUE")
                # Track has changed and is at start of new track
                self.mass.create_task(
                    target=self._advance_queue,
                    eager_start=True,
                    queue_id=self._attr_active_source,
                    task_id=f"ohm_advance_queue_{self.player_id}",
                )

    async def _handle_transport_event(
        self, state_variables: Sequence[UpnpStateVariable[Any]]
    ) -> None:
        """
        Handle state variable changes from Transport service

        :param state_variables: the state variables whose values have changed
        """
        for sv in state_variables:
            match sv.name:
                case TransportState.STREAM_ID:  # indicates track has changed
                    pass
                case TransportState.TRANSPORT_STATE:
                    _playback_state = self._transport_state_to_playback_state(sv.value)
                    if _playback_state is not None:
                        self._attr_playback_state = _playback_state
                case TransportState.REPEAT:
                    self._attr_repeat_state = sv.value
                case TransportState.SHUFFLE:
                    self._attr_shuffle_state = sv.value
                case _:
                    pass

    async def _handle_volume_event(
        self, state_variables: Sequence[UpnpStateVariable[Any]]
    ) -> None:
        """
        Handle state variable changes from Volume service

        :param state_variables: the state variables whose values have changed
        """

        for sv in state_variables:
            match sv.name:
                case VolumeState.MUTE:
                    self._attr_volume_muted = sv.value
                case VolumeState.VOLUME:
                    self._attr_volume_level = (
                        int(sv.value) if sv.value is not None else None
                    )
                case _:
                    pass

    def _handle_event(
        self,
        service: UpnpService,
        state_variables: Sequence[UpnpStateVariable[Any]],
    ) -> None:
        """
        Handle changed state variables value event from Linn/OpenHome Media device.

        For each service, create a corresponding task
        Default is to defer state updates unless called explicitly
        """
        if not state_variables:
            # Indicates a failure of subscription so revert to polling
            # self.logger.debug("Event with no state variables received")
            self.force_poll = True
            return

        # active_queue = self.mass.player_queues.get_active_queue(self.player_id)
        # active_queue_id = active_queue.queue_id if active_queue else None
        # active_queue = self.mass.playerqueues.get_active_queue(self.player_id)
        # if active_queue is None:
        #     return
        # active_queue_id = active_queue.queue_id
        # queue = self.mass.player_queues.get(active_queue_id)

        # Cases not intended to be exhaustive
        match service.service_id:
            # case ServiceId.CREDENTIALS:
            #     pass
            case ServiceId.INFO:
                self.mass.create_task(
                    target=self._handle_info_event(state_variables=state_variables),
                    # task_id=f"ohm_handle_info_event_{self.player_id}",
                )
            # case ServiceId.PINS:
            #     pass
            case ServiceId.PLAYLIST:
                self.mass.create_task(
                    target=self._handle_playlist_event(state_variables=state_variables),
                    # task_id=f"ohm_handle_playlist_event_{self.player_id}",
                )
            case ServiceId.PRODUCT:
                self.mass.create_task(
                    target=self._handle_product_event(state_variables=state_variables),
                    # task_id=f"ohm_handle_product_event_{self.player_id}",
                )
            case ServiceId.RADIO:
                self.mass.create_task(
                    target=self._handle_radio_event(state_variables=state_variables),
                    # task_id=f"ohm_handle_radio_event_{self.player_id}",
                )
            # case ServiceId.RECEIVER:
            #     pass
            # case ServiceId.SENDER:
            #     pass
            case ServiceId.TIME:
                self.mass.create_task(
                    target=self._handle_time_event(state_variables=state_variables),
                    # task_id=f"ohm_handle_time_event_{self.player_id}",
                )
            case ServiceId.TRANSPORT:
                self.mass.create_task(
                    target=self._handle_transport_event(
                        state_variables=state_variables
                    ),
                    # task_id=f"ohm_handle_transport_event_{self.player_id}",
                )
            case ServiceId.VOLUME:
                self.mass.create_task(
                    target=self._handle_volume_event(state_variables=state_variables),
                    # task_id=f"ohm_handle_volume_event_{self.player_id}",
                )
            # case ServiceId.UPDATE:
            #     pass
            case _:
                self.logger.debug(
                    "Unhandled event for service id: %s", service.service_id
                )

        self.last_seen = time.time()

        # create_task does not create a duplicate
        # self.mass.create_task(
        #     self._deferred_update(),
        #     # eager_start=True,
        #     task_id=f"ohm_deferred_update_{self.player_id}",
        # )
        self.update_state()

    async def _update_player(self, poll_first: bool = False) -> None:
        """
        Update Linn/OpenHome Media Player.

        :param poll_first: Refresh the device state before reading it, so that the
            position info belongs to the state that is about to be reported.
        """
        if poll_first:
            # an unavailable device is reported as such by the state update below
            with suppress(PlayerUnavailableError):
                await self.poll()
        prev_url = (
            self._attr_current_media.uri if self._attr_current_media is not None else ""
        )
        prev_state = self.state
        # await self.set_dynamic_attributes()
        current_url = (
            self._attr_current_media.uri if self._attr_current_media is not None else ""
        )
        current_state = self.state

        if (prev_url != current_url) or (prev_state != current_state):
            # fetch track details on state or url change
            self.force_poll = True
        try:
            self.update_state()
        except KeyError, TypeError:
            # at start the update might come faster than the config is initialized
            await asyncio.sleep(2)

        self.update_state()

    def _set_player_features(self) -> None:
        """Set Player Features based on config values and capabilities."""
        assert self.profile is not None  # for type checking
        supported_features: set[PlayerFeature] = set()
        if self.has_play_media:
            supported_features.add(PlayerFeature.PLAY_MEDIA)
            supported_features.add(PlayerFeature.ENQUEUE)
            supported_features.add(PlayerFeature.PAUSE)
            supported_features.add(PlayerFeature.NEXT_PREVIOUS)
        if self.profile.has_product_standby:
            pass  # TODO need to incorporate this sensibly
            # supported_features.add(PlayerFeature.POWER)
        if self.profile.has_transport_seek_second_absolute:
            supported_features.add(PlayerFeature.SEEK)
        if self.profile.has_volume_mute:
            supported_features.add(PlayerFeature.VOLUME_MUTE)
        if self.profile.has_volume_set:
            supported_features.add(PlayerFeature.VOLUME_SET)
        if self.profile.has_product_set_source_index:
            supported_features.add(PlayerFeature.SELECT_SOURCE)

        self._attr_supported_features = supported_features
