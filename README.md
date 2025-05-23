# Federated Learning for Trajectory Prediction

This project demonstrates three federated learning strategies for trajectory prediction using the ETH dataset family (Hotel, Zara1, Zara2, ETH):

- **FedAvg**
- **FedCMB** (Custom Metric-Based)
- **FedRep** (Representation Personalization)

## File Structure

```
src/
  client.py             # Used by FedAvg and FedCMB clients
  client_fedrep.py      # Used by FedRep clients
  server.py             # FedAvg server
  server_fedcmb.py      # FedCMB server
  server_fedrep.py      # FedRep server
```

## Dataset Configuration

The configuration files are located in:

```
config/TP/FlowChain/
```

Each config file corresponds to a scene (e.g., `hotel.yml`, `zara1.yml`, etc.).

## Prerequisites

- Python 3.8+
- Flower >= 1.18.0
- PyTorch, numpy, yacs, tqdm

Install requirements:

```bash
pip install -r requirements.txt
```

## How to Run

You can simulate federated training with four clients using different scenes as local data.

### 1. FedAvg (10 Rounds)

```bash
# Server
python src/server_app.py --num_rounds 10 > server-log-FedAvg.txt 2>&1

# Clients
python src/client_app.py --partition_id 0 --gpu 0 --config_file config/TP/FlowChain/hotel.yml --data_fraction 1.0 > client1-FedAvg.txt 2>&1
python src/client_app.py --partition_id 1 --gpu 0 --config_file config/TP/FlowChain/zara1.yml --data_fraction 1.0 > client2-FedAvg.txt 2>&1
python src/client_app.py --partition_id 2 --gpu 0 --config_file config/TP/FlowChain/zara2.yml --data_fraction 1.0 > client3-FedAvg.txt 2>&1
python src/client_app.py --partition_id 3 --gpu 0 --config_file config/TP/FlowChain/eth.yml --data_fraction 1.0 > client4-FedAvg.txt 2>&1
python src/client_app.py --partition_id 4 --gpu 0 --config_file config/TP/FlowChain/mot.yml --data_fraction 1.0 --visualize > client5-FedAvg.txt 2>&1
```

### 2. FedCMB (10 Rounds)

```bash
# Server
python src/server_fedcmb.py --num_rounds 10 > server-log-FedCMB.txt 2>&1

# Clients (Same as FedAvg)
python src/client.py --config_file config/TP/FlowChain/hotel.yml --data_fraction 1.0 > client1-FedCMB.txt 2>&1
python src/client.py --config_file config/TP/FlowChain/zara1.yml --data_fraction 1.0 > client2-FedCMB.txt 2>&1
python src/client.py --config_file config/TP/FlowChain/zara2.yml --data_fraction 1.0 > client3-FedCMB.txt 2>&1
python src/client.py --config_file config/TP/FlowChain/eth.yml --data_fraction 1.0 > client4-FedCMB.txt 2>&1
```

### 3. FedRep (10 Rounds)

```bash
# Server
python src/server_fedrep.py --num_rounds 10 > server-log-FedRep.txt 2>&1

# Clients
python src/fedrep_client_app.py --partition_id 0 --gpu 0 --config_file config/TP/FlowChain/hotel.yml --data_fraction 1.0 > client1-FedAvg.txt 2>&1
python src/fedrep_client_app.py --partition_id 1 --gpu 0 --config_file config/TP/FlowChain/zara1.yml --data_fraction 1.0 > client2-FedAvg.txt 2>&1
python src/fedrep_client_app.py --partition_id 2 --gpu 0 --config_file config/TP/FlowChain/zara2.yml --data_fraction 1.0 > client3-FedAvg.txt 2>&1
python src/fedrep_client_app.py --partition_id 3 --gpu 0 --config_file config/TP/FlowChain/eth.yml --data_fraction 1.0 > client4-FedAvg.txt 2>&1
python src/fedrep_client_app.py --partition_id 4 --gpu 0 --config_file config/TP/FlowChain/mot.yml --data_fraction 1.0 --visualize > client5-FedAvg.txt 2>&1
```

## Logs & Evaluation

- Logs will be saved to corresponding text files per client/server.
- ADE (Average Displacement Error) and FDE (Final Displacement Error) are computed every round.

## Notes

- Make sure to use separate terminals for each client and the server.
- The `--data_fraction 1.0` option ensures each client uses the full scene data.
- `client_fedrep.py` separates model into representation and head as required by the FedRep strategy.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

This is a research collaboration between the **University of Tennessee at Chattanooga (UTC)** and **DENSO Corporation**.

See [LICENSE](./LICENSE) for full details.
# flowchain-implementation
# flowchain-implementation
# flowchain-implementation
