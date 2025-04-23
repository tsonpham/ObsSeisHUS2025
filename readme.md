# Introduction to Observational Seismology Workshop

<!-- WARNING: This website is under active constructions. Its content might be changed without notice! -->

Facilitator:
- Dr. Phạm Thành Sơn (ANU)

Organizers:
- A/Prof. Phó Đức Tài (HUS)
- A/Prof. Lê Hồng Phương (HUS)
- Dr. Nguyễn Thị Minh Huyền (HUS)
- A/Prof. Trần Thanh Tuấn (HUS)

Venue: VNU Hanoi University of Science, 334 Nguyễn Trãi, Thanh Xuân, Hà Nội.

Time: April 21-25, 2025

Registration form *(deadline April 10)*: https://forms.gle/A7u4UtuP463DfQ1TA

## What do we learn in this workshop
This workshop is a quick introduction to resources and tools for Observational Seismology. The course content covers the following topics:

* Introduction to digital seismic data, including (1) principles of seismometry, (2) seismic data as digital signals, (3) global seismic databases, (4) basic data processing tools. 
* Introduction to geographical mapping skills + visualize scientific data, including (1) Python package to draw a geophysical map of a region, (2) plotting scientific data on a map.
* Introduction to inverse problem theory and classical inverse problems in seismology: (1) earthquake location and (2) seismic moment tensor inversion. 
* Introduction to cross-correlation techniques (i.e., relatively modern seismology technique): theory and example of autocorrelation for shallow Earth imaging.
* A brief introduction to machine learning in seismology: automatic earthquake detection. 

### About the faciliator
[Dr. Phạm](https://www.tsonpham.net/) is an observational seismologist, who uses seismic waves to understand the Earth’s interior structures and seismic energy sources using mathematical tools, such as signal processing, numerical modeling, and geophysical inference. He is particularly interested in structures and processes a few kilometers beneath the surface, such as polar ice sheets, down to the Earth’s deepest shell, including its cores. To date, one of his visible contributions is to help understand better the architecture of the seismic wavefield several hours after large earthquakes and use it to decipher several long-lasting puzzles regarding the Earth’s inner core. In current and near-future research, he aims to expand my seismological toolbox to advance research on the topics, focusing on understanding the structures and dynamics of the polar ice sheets in Antarctica and Greenland in the changing climate. 

## Recommended pre-class reading list
Here I compile a list of some reading materials about some useful tools in observational seismology:
- What's inside the Earth: Interactive poster [link](https://www.earthscope.org/inside-the-earth-poster/)
- Coding environment: Jupyter Notebook [link to tutorial](https://colab.research.google.com/notebooks/intro.ipynb#scrollTo=GJBs_flRovLc)
- Free cloud server: Google Colab [link to tutorial](https://colab.research.google.com/notebooks/intro.ipynb#scrollTo=5fCEDCU_qrC0)
- Basic mapping tool: Basemap [link to tutorial](https://matplotlib.org/basemap/stable/users/geography.html)
- Theoretical travel time and ray paths: Obspy Taup [link to tutorial](https://docs.obspy.org/packages/obspy.taup.html)
- Access to seismic data servers: Obspy FDSN Client [link to tutorial](https://docs.obspy.org/packages/obspy.clients.fdsn.html#module-obspy.clients.fdsn)
- Efficient Bayesian sampler: emcee [link to tutorial](https://emcee.readthedocs.io/en/stable/tutorials/line/)
- Google machine learning crash course [link to tutorial](https://developers.google.com/machine-learning/crash-course/linear-regression)
- Machine Learning Cơ Bản by Vũ Hữu Tiệp [link to ebook](https://github.com/tiepvupsu/ebookMLCB)
- Seisbench: A toolbox for machine learning in seismology [link](https://seisbench.readthedocs.io/en/stable/)

More up-to-date reading list can be found in this [Google Docs](https://docs.google.com/document/d/1o0qxpMSplIhqRjPVi1Lvl_tMI04KH6avsbPa5C-PZC4/edit?usp=drive_link).

## In-class activities

*Module 1: Introduction, geographical mapping*
<!-- * [Notes](Day1/notes.md) -->
* Lecture slides [PDF](https://drive.google.com/file/d/1KUkcooDYMBD843-AA1lpRnQ8i88LBAO_/view?usp=sharing) [PPTX](https://anu365-my.sharepoint.com/:p:/r/personal/u5883665_anu_edu_au/Documents/HUS2025/Module1.pptx?d=w397b024a68eb4e7f817f38c052061bf5&csf=1&web=1&e=Io1eim)
* In-class exercise: Plotting Maps: Seismograph and Seismicity in Vietnam [![Open In Colab](https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670)](https://colab.research.google.com/github/tsonpham/ObsSeisHUS2025/blob/master/Day1/D1_Lab.ipynb)
* Self-practice exercise: Exploring seismic stations in Antarctica [![Open In Colab](https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670)](https://colab.research.google.com/github/tsonpham/ObsSeisHUS2025/blob/master/Day1/D1_Prac.ipynb)

*Module 2: Ray theory, seismometry, seismic databases*
<!-- * [Overview](Day2/notes.md) -->
* Lecture slides [PPTX (download to view)](https://docs.google.com/presentation/d/1a3Esj2lMUnZUAi9Tt2DM-P9fcKZ78ww8/edit?usp=drive_link&ouid=115613731196113719130&rtpof=true&sd=true)
* In-class exercise: Ray theoretical travel times and paths
 [![Open In Colab](https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670)](https://colab.research.google.com/github/tsonpham/ObsSeisHUS2025/blob/master/Day2/D2_Lab.ipynb)
* Self-practice exercise: Triangulation of M5.2 Kon Tum 28/07/2024 earthquake [![Open In Colab](https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670)](https://colab.research.google.com/github/tsonpham/ObsSeisHUS2025/blob/master/Day2/D2_Prac.ipynb)

*Module 3: Geophysical inverse problem*
<!-- * [Notes](Day3/notes.md) -->
* Lecture slides [PPTX (download to view)](https://docs.google.com/presentation/d/1aSIX8qNMxjs41evKr-xIcPzySY_q-wlB/edit?usp=drive_link&ouid=115613731196113719130&rtpof=true&sd=true)
* In-class exercise: Linear regression [![Open In Colab](https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670)](https://colab.research.google.com/github/tsonpham/ObsSeisHUS2025/blob/master/Day3/D3_Lab.ipynb)
* Self-practice exercise: Earthquake location as an inverse problem [![Open In Colab](https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670)](https://colab.research.google.com/github/tsonpham/ObsSeisHUS2025/blob/master/Day3/D3_Prac.ipynb)
* Advanced exercise: Seismic moment tensor inversion (by Julien Thurin) [![Open In Colab](https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670)](https://colab.research.google.com/drive/1UJWOompBz9MlJN0B6SoVKzF8Whz_1nPp?usp=sharing#scrollTo=n8Gxw3DPkxAb)

*Module 4: Seismic interferometry and study of the Earth’s structures*
<!-- * [Notes](Day4/notes.md) -->
* Lecture slides [PPTX (download to view)](https://docs.google.com/presentation/d/1bWzEXppb5zZ2RlFcG0IK9pZA0-Gv71w2/edit?usp=drive_link&ouid=115613731196113719130&rtpof=true&sd=true)
* In-class exercise: Teleseismic *P*-wave coda autocorrelation [![Open In Colab](https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670)](https://colab.research.google.com/github/tsonpham/ObsSeisHUS2025/blob/master/Day4/D4_Lab.ipynb)
* Self-practice exercise: Single-event global correlogram [![Open In Colab](https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670)](https://colab.research.google.com/github/tsonpham/ObsSeisHUS2025/blob/master/Day4/D4_Prac.ipynb)

*Module 5: Machine learning in Seismology*
<!-- * [Notes](Day5/notes.md) -->
* Lecture slides [PPTX (download to view)]()
* In-class exercise: Convolutional neural network for PKIKP onset phase picker [![Open In Colab](https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670)](https://colab.research.google.com/github/tsonpham/ObsSeisHUS2025/blob/master/Day5/D5_Lab.ipynb)
* Self-practice exercise: Introduction to [seisbench](https://seisbench.readthedocs.io/en/stable/index.html): A toolbox for machine learning in seismology [![Open In Colab](https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670)](https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/01b_model_api.ipynb)