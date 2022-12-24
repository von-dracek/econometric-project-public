import gc
import logging
from copy import deepcopy
from multiprocessing import Process
from typing import Any, Dict

import pandas as pd
from kedro.io import PartitionedDataSet
from mpire import WorkerPool

log = logging.getLogger(__name__)


class AdjustedPartitionedDataset(PartitionedDataSet):
    def _save_partition(self, partition_data, partition_id):
        kwargs = deepcopy(self._dataset_config)
        partition = self._partition_to_path(partition_id)
        # join the protocol back since tools like PySpark may rely on it
        kwargs[self._filepath_arg] = self._join_protocol(partition)
        dataset = self._dataset_type(**kwargs)  # type: ignore
        if callable(partition_data):
            try:
                partition_data = partition_data()
            except Exception as e:
                log.error(f"Error when saving dataset {partition_id}", exc_info=e)
            if (partition_data is not None) and not (callable(partition_data)):
                if isinstance(partition_data, pd.DataFrame):
                    if (
                        not partition_data.empty
                    ):  # it may happen that the processing function returns empty dataframe
                        dataset.save(partition_data)
                    else:
                        log.error(f"Error when saving dataset {partition_id}. Dataframe empty.")
                else:
                    dataset.save(partition_data)
        else:
            dataset.save(partition_data)
        gc.collect()

    def _save(self, data: Dict[str, Any]) -> None:
        if self._overwrite and self._filesystem.exists(self._normalized_path):
            self._filesystem.rm(self._normalized_path, recursive=True)
        n_jobs = 6  # TODO: parametrize this from command line
        data = [
            (partition_data, partition_id) for partition_id, partition_data in sorted(data.items())
        ]

        with WorkerPool(n_jobs=n_jobs, use_dill=True, daemon=False) as pool:
            pool.map(self.__save_partition_in_subprocess, data, progress_bar=True)

        self._invalidate_caches()

    def __save_partition_in_subprocess(self, partition_data, partition_id):
        # this had to be done this way since otherwise, the dataset is somehow stored in
        # memory and not thrown away, leading to OOM errors
        p = Process(target=self._save_partition, args=(partition_data, partition_id))
        p.start()
        p.join()
        p.kill()
