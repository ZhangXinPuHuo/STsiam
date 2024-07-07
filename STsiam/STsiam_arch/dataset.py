import os
import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
from torch.utils.data import Dataset
import random




def load_pkl(pickle_file: str) -> object:
    """Load pickle data.

    Args:
        pickle_file (str): file path

    Returns:
        object: loaded objected
    """

    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


#由basicts的TimeSeriesForecastingDataset改写而来
class TimeSeriesForecastingDataset(Dataset):
    """Time series forecasting dataset."""

    def __init__(self, data_file_path: str, index_file_path: str, mode: str) -> None:
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path)
        # read raw data (normalized)
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        self.data = torch.from_numpy(processed_data).float()
        # read index
        self.index = load_pkl(index_file_path)[mode]
        self.mode = mode
        self.segment_start_indices = {
            'train': 0,
            'valid': round(len(processed_data) * 0.6),  # Assuming train_ratio is 0.6
            'test': round(len(processed_data) * (0.6 + 0.2))  # Assuming valid_ratio is 0.2
        }

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("BasicTS can not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("BasicTS can not find index file {0}".format(index_file_path))

    def __getitem__(self, index: int) -> tuple:
        """Get a sample and the distance between the end of history data and the start of future data.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data, distance), where the shape of each is L x N x C,
                   and distance is the number of time steps between the end of history data and the start of future data.
        """
        idx = list(self.index[index])
        segment_start_index = self.get_segment_start_index()
        if isinstance(idx[0], int):
            # Future data remains the same
            future_data = self.data[idx[1]:idx[2]]

            # Determine the range for random history data
            # Ensure it does not overlap with the future data range
            if idx[1] > 0:  # Check if there is space before future data
                random_history_end = idx[1]  # End of history data is start of future data
                random_history_start = random.randint(segment_start_index, random_history_end-(idx[2]-idx[1]))  # Random start from available history
                history_data = self.data[random_history_start:random_history_start+(idx[2]-idx[1])]
                # Calculate distance as the gap between the end of history data and the start of future data
                distance = idx[1] - random_history_start
            else:
                # No available history data before the future data, return empty or handle as needed
                history_data = torch.empty(0)  # or handle appropriately
                distance = 0  # No gap if there's no history data
        else:
            # Handling for non-integer indices (if necessary)
            raise NotImplementedError("Handling of non-integer or custom indices is not implemented.")

        return future_data, history_data, distance

    def __len__(self):
        """Dataset length

        Returns:
            int: dataset length
        """

        return len(self.index)
    def get_segment_start_index(self):
        """Retrieve the start index of the current segment."""
        return self.segment_start_indices[self.mode]
    
    