# Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Acknowledgement](#acknowledgement)
- [License](#license)
# Overview

Open-Personal-Dosimeter is a fully open spectral dosimeter and a build tutorial. Two setups are proposed (1) A setup with a low-cost AS7341 sensor and (2) a higher-end setup with NanoLambda sensor. Both of these two include a guide to wiring and making the 3D enclusure of the assembly, as well as a mobile data monitoring/storage pipeline with MQTT communication protocol.

## Repository structure

- `hardware/` — List of parts and components, as used in this work, wiring diagrams, guide on printing the enclosure.
- `firmware/` — embedded software for data acquisition and device control (if any)
- `software/` — host-side tools for data transfer | explanation of the communication protocols
- `data/` — example datasets and data format references (if any)
- `docs/` — other project documentation and development notes (if any)

## Acknowledgement

This project is funded by the **TU Delft Open Science Foundation**.

## License

This repository is licensed under the **GNU General Public License v3.0**.
