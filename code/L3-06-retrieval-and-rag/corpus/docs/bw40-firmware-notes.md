# BW-40 Firmware Release Notes

Effective: 1 July 2025. Status: current. Owner: Product team.

## Version 3.3

Adds support for the optional LoRa radio module, which sends readings to a gateway up to 5 kilometres away in open country. The default logging interval for new installations changes to 10 minutes; existing loggers keep their current setting after the update. Update through the Brindle Field app; the update takes about four minutes and must not be interrupted.

## Version 3.2

Fixes a clock drift problem in which the internal clock could lose up to two minutes per month in very cold conditions. All loggers deployed in cold regions should be updated.

## Version 3.1

Adds overwrite mode for full memory and the ability to export readings as CSV directly from the app.

## Error codes

Error codes appear in the app next to the logger name.

- E-12: sensor disconnected. Check that the probe cable is fully pushed in and the connector is dry.
- E-31: reading out of range. Usually caused by a probe that is not in contact with soil; reinstall the probe with firm contact.
- E-47: memory full. Download the readings and clear the memory, or turn on overwrite mode.
- E-58: firmware update failed. Leave the logger within Bluetooth range and run the update again; the logger keeps the previous firmware until an update succeeds.

## Known issues

On some Android phones the app loses the Bluetooth connection when the screen turns off during a download. Keep the screen on until the download completes.
