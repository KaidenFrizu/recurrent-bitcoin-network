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
- TensorFlow 2.8.0 or newer
- Other packages listed in [requirements.txt](requirements.txt)

### Optional
- Messari API Key. This can be acquired through [Messari](https://messari.io/) to be used in their [web API](https://messari.io/api/docs).

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

5. Run the notebooks `Documentation.ipynb` and `Modelling.ipynb`. These are found in [docs](docs) folder.

---

### Contributing

To contribute, see the [contributing guidelines](CONTRIBUTING.md).
