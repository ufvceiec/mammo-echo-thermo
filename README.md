<br />
<p align="center">
  <a href="https://github.com/ufvceiec/mammo-echo-thermo">
    <img src="https://i.imgur.com/mgc7iqA.png" alt="Logo" width="100" height="100">
  </a>

  <h3 align="center">Mammo Echo Thermo</h3>

  <p align="center">
    Breast cancer detection from thermal imaging
    <br />
    <a href="http://www.ceiec.es/investigacion/proyectos/"><strong>Read more about the study »</strong></a>
    <br />
    <br />
    <a href="https://github.com/ufvceiec/mammo-echo-thermo/">View Code</a>
    ·
    <a href="https://github.com/ufvceiec/mammo-echo-thermo/issues">Report Bug</a>
    ·
    <a href="https://github.com/ufvceiec/mammo-echo-thermo/discussions">Start a discussion</a>
  </p>
</p>
<br />

<p align="center">
    <a href="https://github.com/ufvceiec/mammo-echo-thermo" alt="Github downloads">
        <img src="https://img.shields.io/github/downloads/ufvceiec/mammo-echo-thermo/total?logo=github&style=flat-square" />
    </a>
    <a href="https://github.com/ufvceiec/mammo-echo-thermo/issues" alt="Github open issues">
        <img src="https://img.shields.io/github/issues-raw/ufvceiec/mammo-echo-thermo?logo=github&style=flat-square" />
    </a>
    <a href="https://github.com/ufvceiec/mammo-echo-thermo/issues" alt="Github clossed issues">
        <img src="https://img.shields.io/github/issues-closed-raw/ufvceiec/mammo-echo-thermo?logo=github&style=flat-square" />
    </a>
    <a href="https://github.com/ufvceiec/mammo-echo-thermo/releases" alt="Github releases">
        <img src="https://img.shields.io/github/v/release/ufvceiec/mammo-echo-thermo?logo=github&style=flat-square" />
    </a>
    <a href="https://github.com/ufvceiec/mammo-echo-thermo/commits" alt="Github commit activity">
        <img src="https://img.shields.io/github/commit-activity/y/ufvceiec/mammo-echo-thermo?logo=github&style=flat-square" />
    </a>
</p>

The main purpose of this project is to develop a comprehensive decision support system for breast cancer screening.

This program has the following objectives:
- Determine the optimal screening pattern for each woman based on her age and risk factors such as family history, breast density, lifestyle, contraceptive use, hormone replacement therapy, etc. The pattern will indicate which tests should be performed, in what order and with what frequency, with cost-effectiveness criteria to determine, in case of suspicious indications, the most appropriate diagnostic tests.
- To demonstrate by means of a rigorous study that nowadays thermography has sufficient precision to be used in breast cancer screening, in combination with mammography and ultrasound, providing important advantages.

This study will collect all its data from two hospitals: [HM Montepríncipe](https://www.hmmonteprincipe.com/) (Boadilla del Monte, Madrid) and [Holy Spirit Hospital](https://www.holyspirithospital.org/) (Makeni, Sierra Leone). The former has the most advanced screening and diagnostic facilities, while the latter lacks the means to perform mammography, so it will only be able to perform ultrasound and thermography.

## Installation
To run this project it is necessary to download this repository. It can be done under the command:
```
git clone https://github.com/ufvceiec/mammo-echo-thermo.git
```

Once the repository is cloned on your local machine, it will be necessary to have Docker installed (if you want to use Docker with GPUs support, use the [the following installation guide](https://github.com/FernandoPerezLara/docker-tensorflow-gpu)).

To execute the program, run the following command:
```
cd mammo-echo-thermo
docker-compose up -d
```

And then enter inside the container and run the file `main.py`.
