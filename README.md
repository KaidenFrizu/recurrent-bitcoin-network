# Recurrent Bitcoin Network

[![CodeFactor](https://www.codefactor.io/repository/github/kaidenfrizu/recurrent-bitcoin-network/badge)](https://www.codefactor.io/repository/github/kaidenfrizu/recurrent-bitcoin-network) [![DeepSource](https://deepsource.io/gh/KaidenFrizu/recurrent-bitcoin-network.svg/?label=resolved+issues&token=hwJ9eS-xya6xRz48SvXNyMUL)](https://deepsource.io/gh/KaidenFrizu/recurrent-bitcoin-network/?ref=repository-badge)

### About

This repository is part of our Data Science Undergraduate Thesis. It contains the source code for implementing Bitcoin price prediciton using a Seq2Seq RNN architecture via TensorFlow.

The data was collected through [Messari](https://messari.io/) API. The documentation can be found [here](https://messari.io/api/docs).

---

### Disclaimer

This project is only used for academic purposes, with no implications on its feasibility in actual price forecasting. We are not accountable for any loss or failures in your investments for any implementations of this project. This is NOT financial advice. For details, refer to the [GNU General Public License v3.0](LICENSE).

---

### Prerequesites

- Python 3.7 or newer
- Python packages listed in [requirements.txt](requirements.txt)

---

### How to reproduce this project?

You can reproduce this project through these general instructions.

1. Clone (or download) this repository.
2. Open a terminal and set its current directory to your local repository.
3. Create a virtual environment (`venv`) through the terminal. One example would be creating one name `.env` shown below:

```ps
python -m venv .env
```

4. Afterwards, install the packages found in `requirements.txt`. The most efficient way to install these packages is through the terminal with the following command. This would take a while as it has approximately 400MB download size.

```ps
pip install -r requirements.txt
```

*Chapter Unfinished. Please come back later.*

---

### Contributing

Here are some contributing guideline on submitting an issue or opening a pull request.

- Submit an issue
  
  If you wanted to raise an issue such as problems or suggestions, you may create one in this repository. However, certain guidelines must be considered before raising one.

  1. Make sure that the issue is not a duplicate of a previous issue. If there is an existing one, there's no need to create another issue. You may add reactions to such issue instead if you wanted to support in resolving as such.
  2. Include as many relevant details as possible. Add code snippets and avoid inserting images, if possible. Show error messages if necessary.

- Pull request

  *Since this is an academic research, we are very conservative on the proposed features or changes on your pull requests. We may not accept such changes in our own discretion in adherence to our university's thesis guidelines.*
  
  Before opening a pull request, here are some general guidelines that is advised to avoid conflicts.

  1. The pull request would come from your fork. Pull requests coming from a clone would be closed immediately.
  2. Your pull request should be from a feature branch from `main` **in your forked repository**. Same action as stated previously would be given otherwise.
  3. Include general details of your pull request: What is aimed to fix, what features would be added, etc.
  4. Make sure your code adheres to the coding standards present in this repository (to be added soon).
