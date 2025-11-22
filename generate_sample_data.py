import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_network_traffic_data(n_samples=1000, anomaly_ratio=0.1):
    """
    Generate synthetic network traffic data with normal and anomalous patterns.
    
    Features include:
    - packet_size: Size of network packet in bytes
    - duration: Connection duration in seconds
    - src_bytes: Bytes sent from source
    - dst_bytes: Bytes sent to destination
    - protocol_type: Protocol used (0=TCP, 1=UDP, 2=ICMP)
    - flag: Connection flag status (0-5)
    - src_port: Source port number
    - dst_port: Destination port number
    - packet_rate: Packets per second
    - error_rate: Percentage of error packets
    """
    
    np.random.seed(42)
    
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies
    
    # Generate normal traffic
    normal_data = {
        'timestamp': [datetime.now() - timedelta(seconds=i*10) for i in range(n_normal)],
        'packet_size': np.random.normal(512, 100, n_normal).clip(64, 1500),
        'duration': np.random.exponential(2, n_normal).clip(0.1, 30),
        'src_bytes': np.random.normal(5000, 1500, n_normal).clip(100, 50000),
        'dst_bytes': np.random.normal(5000, 1500, n_normal).clip(100, 50000),
        'protocol_type': np.random.choice([0, 1, 2], n_normal, p=[0.7, 0.25, 0.05]),
        'flag': np.random.choice([0, 1, 2, 3, 4, 5], n_normal, p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02]),
        'src_port': np.random.randint(1024, 65535, n_normal),
        'dst_port': np.random.choice([80, 443, 22, 21, 25, 53], n_normal, p=[0.4, 0.3, 0.15, 0.05, 0.05, 0.05]),
        'packet_rate': np.random.normal(50, 15, n_normal).clip(1, 200),
        'error_rate': np.random.normal(0.5, 0.3, n_normal).clip(0, 5),
    }
    
    # Generate anomalous traffic (unusual patterns)
    anomaly_data = {
        'timestamp': [datetime.now() - timedelta(seconds=i*10) for i in range(n_normal, n_samples)],
        'packet_size': np.concatenate([
            np.random.normal(1400, 50, n_anomalies//3).clip(1200, 1500),  # Large packets
            np.random.normal(100, 20, n_anomalies//3).clip(64, 200),      # Tiny packets
            np.random.normal(512, 100, n_anomalies - 2*(n_anomalies//3)).clip(64, 1500)
        ]),
        'duration': np.concatenate([
            np.random.exponential(15, n_anomalies//2).clip(10, 100),  # Long connections
            np.random.exponential(0.1, n_anomalies - n_anomalies//2).clip(0.01, 0.5)  # Very short
        ]),
        'src_bytes': np.concatenate([
            np.random.normal(40000, 10000, n_anomalies//2).clip(30000, 100000),  # Large transfers
            np.random.normal(200, 100, n_anomalies - n_anomalies//2).clip(10, 500)  # Tiny transfers
        ]),
        'dst_bytes': np.concatenate([
            np.random.normal(40000, 10000, n_anomalies//2).clip(30000, 100000),
            np.random.normal(200, 100, n_anomalies - n_anomalies//2).clip(10, 500)
        ]),
        'protocol_type': np.random.choice([0, 1, 2], n_anomalies, p=[0.3, 0.3, 0.4]),  # More ICMP
        'flag': np.random.choice([0, 1, 2, 3, 4, 5], n_anomalies, p=[0.1, 0.1, 0.2, 0.3, 0.2, 0.1]),  # Unusual flags
        'src_port': np.random.randint(1, 1024, n_anomalies),  # Privileged ports (suspicious)
        'dst_port': np.random.randint(8000, 65535, n_anomalies),  # Unusual destination ports
        'packet_rate': np.concatenate([
            np.random.normal(500, 100, n_anomalies//2).clip(300, 1000),  # Very high rate
            np.random.normal(2, 1, n_anomalies - n_anomalies//2).clip(0.1, 5)  # Very low rate
        ]),
        'error_rate': np.random.normal(15, 5, n_anomalies).clip(10, 50),  # High error rate
    }
    
    # Combine normal and anomalous data
    df_normal = pd.DataFrame(normal_data)
    df_anomaly = pd.DataFrame(anomaly_data)
    
    df_normal['label'] = 0  # Normal
    df_anomaly['label'] = 1  # Anomaly
    
    # Combine and shuffle
    df = pd.concat([df_normal, df_anomaly], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Generate sample dataset
    df = generate_network_traffic_data(n_samples=1000, anomaly_ratio=0.15)
    df.to_csv('sample_network_traffic.csv', index=False)
    print(f"Generated {len(df)} network traffic samples")
    print(f"Normal samples: {sum(df['label'] == 0)}")
    print(f"Anomaly samples: {sum(df['label'] == 1)}")
    print(f"\nDataset saved to 'sample_network_traffic.csv'")
