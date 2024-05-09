# UPenn CIS6600 Authoring Tool *Terroder*
- **Author**: Andrew Ding, Xinran Tao
- **Semester**: Spring 2024

## Introduction
The development of _Terroder_, a terrain creation tool, is a response to a distinct need within the realm of digital environment design. Inspired by the work of [Schott et al.](https://dl.acm.org/doi/10.1145/3592787), Terroder is aimed at addressing three key criteria often overlooked by existing terrain generation tools: producing realistic and detailed output at a micro-scale, allowing for nuanced user control, and maximizing efficiency in terms of user effort. The prevalent terrain generation tools frequently fall short in one or more of these areas, either by limiting the level of detail achievable, restricting user control, or necessitating extensive manual effort for detail refinement.

## Dependencies
Since _Terroder_ utilizes the powerful Python libraries, there are some packages that need to be installed before _Terroder_ can be successfully loaded in Maya. These packages are listed below:
- numpy
- pillow
- opencv-python

You can install Python packages in Maya using the following command in a terminal, assuming that you have correctly added `mayapy` as your system command:
```bash
mayapy -m pip install <flags> <package>
```
For more details about how to install Python packages in Maya, please refer to the official [website](https://help.autodesk.com/view/MAYAUL/2024/ENU/?guid=GUID-72A245EC-CDB4-46AB-BEE0-4BBBF9791627).

