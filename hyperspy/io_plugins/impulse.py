import numpy as np
import os
from datetime import datetime as dt
import csv
import warnings
import logging


# Plugin characteristics
# ----------------------
format_name = "Impulse"
description = "Reads DENSsolutions Impulse log files"
full_support = False
# Recognised file extension
file_extensions = ["csv", "CSV", "impulse"]
default_extension = 0
# Reading capabilities
reads_images = False
reads_spectrum = False
reads_spectrum_image = False
# Writing capabilities
writes = False
non_uniform_axis = False
# ----------------------


_logger = logging.getLogger(__name__)


# At some point, if there is another readerw, whith also use csv file, it will
# be necessary to mention the other reader in this message (and to add an
# argument in the load function to specify the correct reader)
invalid_file_error = (
    "The csv reader can't import the file, please"
    " make sure, this is a valid Impulse log file."
)
invalid_filenaming_error = {
    "No _Metadata file detected in filepath, please"
    " make sure that the Metadata file is included."
}
no_metadata_file_error = {
    "No _Metadata file detected in filepath, please"
    " make sure that the Metadata file is included."
}


def file_reader(filename, *args, **kwds):
    csv_file = impulseCSV(filename)
    return _impulseCSV_log_reader(csv_file)


def _impulseCSV_log_reader(csv_file):
    csvs = []
    for key in csv_file.logged_quantity_name_list:
        try:
            csvs.append(csv_file.get_dictionary(key))
        except BaseException:
            raise IOError(invalid_file_error)
    return csvs


class impulseCSV:
    def __init__(self, filename):
        self.filename = filename
        self._parse_header()
        self._read_data()

    def _parse_header(self):
        with open(self.filename, "r") as f:
            s = f.readline()
            self.column_names = s.strip().split(",")
            if not self._is_impulse_csv_file():
                raise IOError(invalid_file_error)
            self._read_metadatafile()
        self.logged_quantity_name_list = self.column_names[2:]

    def _is_impulse_csv_file(self):
        if "TimeStamp" in self.column_names and len(self.column_names) >= 3:
            return True
        else:
            return False

    def get_dictionary(self, quantity):
        return {
            "data": self._data_dictionary[quantity],
            "axes": self._get_axes(),
            "metadata": self._get_metadata(quantity),
            "original_metadata": {"Impulse_header": self.original_metadata},
        }

    def _get_metadata(self, quantity):
        return {
            "General": {
                "original_filename": os.path.split(self.filename)[1],
                "title": "%s" % quantity,
                "date": self.original_metadata["Experiment date"],
                "time": self.original_metadata["Experiment time"],
            },
            "Signal": {"signal_type": "", "quantity": self._parse_quantity_units(quantity)},
        }


    def _parse_quantity_units(self, quantity):
        quantity_split = quantity.strip().split(" ")
        if len(quantity_split) >1:
            if quantity_split[-1][0]=="(" and quantity_split[-1][-1]==")":
                return quantity_split[-1]
            else: return "NA"
        else: return "NA"
        
    def _get_metadata_time_axis(self):
        return {"value": self.time_axis, "units": self.time_units}

    def _read_data(self):
        names = [
            name.replace(" ", "_").replace("°C","C").replace("(","").replace(")","").replace("/","_").replace("%","Perc")
            for name in self.column_names
        ]
        data = np.genfromtxt(
            self.filename,
            delimiter=",",
            dtype=None,
            names=names,
            skip_header=1,
            encoding="latin1",
        )
        self._data_dictionary = dict()
        for i, name, name_dtype in zip(range(len(names)), self.column_names, names):
            if name == "Experiment time":
                self.time_axis = data[name_dtype]
            elif name == "MixValve":
                mixvalvedatachanged = data[name_dtype]
                for index, item in enumerate(data[name_dtype]):
                    mixvalvedatachanged[index] = (
                        int(int(item.split(";")[0]) + 2) * 100
                        + (int(item.split(";")[1]) + 2) * 10
                        + (int(item.split(";")[2]) + 2)
                    )
                mixvalvedatachangedint = np.array(mixvalvedatachanged, dtype=np.int32)
                self._data_dictionary[name] = mixvalvedatachangedint
            else:
                self._data_dictionary[name] = data[name_dtype]

    def _read_metadatafile(self):
        # Locate the experiment metadata file
        self.original_metadata = {}
        notes = []
        notes_section = False

        if "_Synchronized data" in str(self.filename) or "raw" in str(
            self.filename
        ):  # Check if Impulse filename formatting is intact
            metadata_file = (
                "_".join(str(self.filename).split("_")[:-1]) + "_Metadata.log"
            ).replace("\\", "/")
            if os.path.isfile(metadata_file):
                with open(metadata_file, newline="") as csvfile:
                    metadata_file_reader = csv.reader(csvfile, delimiter=",")
                    for row in metadata_file_reader:
                        if notes_section:
                            notes.append(row[0])
                        elif row[0] == "Live notes":
                            notes_section = True
                            notes = [row[1].strip()]
                        else:
                            self.original_metadata[row[0]] = row[1].strip()
                    self.original_metadata["Notes"] = notes
                    self.start_datetime = np.datetime64(
                        dt.strptime(
                            self.original_metadata["Experiment date"]
                            + self.original_metadata["Experiment time"],
                            "%d-%m-%Y%H:%M:%S",
                        )
                    )

            else:
                raise IOError(no_metadata_file_error)
        else:
            raise IOError(invalid_filenaming_error)

    def _get_axes(self):
        scale = np.diff(self.time_axis[1:-1]).mean()
        units = "s"
        offset = 0
        self.time_units = "Seconds"
        return [
            {
                "size": self.time_axis.shape[0],
                "index_in_array": 0,
                "name": "Time",
                "scale": scale,
                "offset": offset,
                "units": units,
                "navigate": False,
            }
        ]
