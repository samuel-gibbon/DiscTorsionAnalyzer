<p align="center">
  <a href="https://example.com/">
    <img src="/home/sam/Dropbox/PROJECTS/DiscTorsionAnalyzer/LOGO.png" alt="DiscTorsionAnalyzer" width=500 height=200>
  </a>

  <h3 align="center">DiscTorsionAnalyzer</h3>

  <p align="center">
    Software to automatically measure optic disc torsion in colour fundus photographs of the retina
    <br>
    <a href="https://reponame/issues/new?template=bug.md">Report bug</a>
    ·
    <a href="https://reponame/issues/new?template=feature.md&labels=feature">Request feature</a>
  </p>
</p>


## Table of contents

- [Quick start](#quick-start)
- [Status](#status)
- [Bugs and feature requests](#bugs-and-feature-requests)
- [Contributing](#contributing)
- [Creators](#creators)
- [Copyright and license](#copyright-and-license)


## Quick start

### Requirements

1. Linux OS
2. Miniconda
3. GPU

### Installation

Step 1: create virtual environment
```bash
conda update conda
conda create -n DiscTorsionAnalyzer python=2.12 -y
```

Step 2: activate virtual environment and clone repo
```bash
conda activate DiscTorsionAnalyzer
git clone https://github.com/samuel-gibbon/DiscTorsionAnalyzer.git
cd DiscTorsionAnalyzer
```

Step 3: install torch
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
If this step fails, go to https://pytorch.org/get-started/locally/ and follow installation instructions.

Step 4: install other packages
```bash
pip install -r requirements.txt
```
### Running

Open the **config.ini** file and set **pathIn** and **pathOut** <br> 
You can also disable **writeImages** and/or **writeResults** if needed.

```bash
python main.py
```

## Status

Software is currently working for colour fundus images of any size.

## Bugs and feature requests

Have a bug or a feature request? Please first read the [issue guidelines](https://github.com/samuel-gibbon/DiscTorsionAnalyzer/blob/main/CONTRIBUTING.md) and search for existing and closed issues. If your problem or idea is not addressed yet, [please open a new issue](https://github.com/samuel-gibbon/DiscTorsionAnalyzer/issues/new).

## Contributing

Please read through our [contributing guidelines](https://reponame/blob/master/CONTRIBUTING.md). Included are directions for opening issues, coding standards, and notes on development.

Moreover, all HTML and CSS should conform to the [Code Guide](https://github.com/mdo/code-guide), maintained by [Main author](https://github.com/usernamemainauthor).

Editor preferences are available in the [editor config](https://reponame/blob/master/.editorconfig) for easy use in common text editors. Read more and download plugins at <https://editorconfig.org/>.

## Creators

**Creator 1**

- <https://github.com/samuel-gibbon/DiscTorsionAnalyzer.git>

## Copyright and license

Code and documentation copyright 2023-2024 the authors. Code released under the [MIT License](https://reponame/blob/master/LICENSE).

Enjoy :metal:
