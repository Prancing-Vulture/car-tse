# P0 tasks: test the complete in-car voice assistant

## Goal

Find out whether the current assistant can hear the enrolled driver, recognize “Marvin,” understand the command that follows, and avoid acting on passengers or background speech.

## Context

Use the **current `car-tse` repository**, including its existing code and checkpoints. Before starting, fetch the latest Git state and work on a separate branch so the changes can be reviewed. The repository contains a multi-stage voice assistant and public speech and cabin-noise data. However, it does not currently contain a suitable continuous recording where the same driver provides enrollment audio, says the wake word, and then gives a command. Existing cross-dataset or synthetic results can help debug the system, but they do not prove real-driver performance.

## Three tasks

### 1. Check and prepare a small real-driver test set

- First check whether suitable recordings already exist.
- If not, record or obtain permitted recordings where the same driver provides enrollment audio, says “Marvin,” and gives a command.
- Include passenger speech, a passenger saying “Marvin,” and periods with no intended command.
- Record the vehicle and noise conditions, and mark when wake words and commands occur.
- Keep the people or sessions used for final testing separate from those used to tune settings.
- Use recordings only when their use is permitted. If recording access is unavailable, document the limitation rather than presenting synthetic or cross-dataset examples as real-driver evidence.

### 2. Evaluate the whole assistant

- Run the current full pipeline and checkpoints on the prepared recordings.
- Adjust settings using only the tuning data, then freeze them before the final test.
- Report how often the driver's wake word is detected, how often the command is understood correctly, how often the system acts when it should not, and how long it takes to respond.
- Show results for different noise and passenger conditions, and include the number of drivers and recording hours.

### 3. Compare the full system with simpler versions

Use the same held-out recordings to compare the full system against versions with one component disabled at a time: speaker extraction, presence checking, speaker matching, and wake-word smoothing. Also compare with recognizing commands without the wake-word steps. Save the settings, run commands, and results so another person can repeat the comparisons.

## What success means

The study should show whether the complete assistant works in driver/passenger situations, which components help, and what still fails. The experiments may improve the system, but better scores are not guaranteed. Stronger real-driver evidence could make a future paper more convincing; publication is not guaranteed and depends on the data quality, study size, and results.

Start with a small pilot to make sure recording and evaluation work. Clearly describe the pilot's size and do not claim broad real-world reliability from a small sample.

## Resources needed

- Access to this repository and its checkpoints.
- Suitable cabin recordings or access to a car for recording.
- At least one consenting speaker; several drivers are preferable.
- A microphone setup compatible with the system.
- Time to label the recordings and enough disk space to store audio and results.

## Deliverables

1. A short data and recording protocol, including permissions and limitations.
2. The prepared data split and annotations, or a clear explanation of why suitable recordings could not be obtained.
3. Reproducible run commands and settings, frozen test metrics, and component comparison results.
4. A concise summary of what worked, what failed, and what should be done next.
