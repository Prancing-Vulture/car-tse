# Capture hardware preflight

Date: 2026-10-12 (rechecked)

## Result

The read-only device query from `scripts/data/record_real_driver_session.py
--list-devices` found the following usable input endpoints on the host:

| Device | Exposed input channels |
|---|---:|
| Microsoft Sound Mapper - Input | 2 |
| Microphone (Fifine Microphone), MME | 1 |
| Microphone (Fifine Microphone), Windows DirectSound | 1 |
| Microphone (Fifine Microphone), Windows WASAPI | 2 |
| Microphone (Fifine Microphone), WDM-KS | 1 |

No four-channel cabin array was exposed. The current production configuration
expects 4-channel, 16 kHz input, so no real-driver recording was started. A
future mono/stereo recording could be used only as exploratory evidence after
explicit consent and must not be used to satisfy the production hardware gate.

No audio or participant data was created by this preflight.

The recorder now performs the same contract check before it creates a session
directory. A four-channel request against the available Fifine input was
rejected with no directory created. Successful future captures will also
record the resolved device capability in `session_metadata.json`.
