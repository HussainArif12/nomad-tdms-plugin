import logging
import os

from nomad.datamodel import EntryArchive

from nomad_tdms_plugin.parsers.parser import NewParser


def test_parse_file():
    parser = NewParser()
    archive = EntryArchive()
    filename = "PROCESS_DATA_STORAGE_2025-12-19_12-30-09"
    base_dir = "tests/example_uploads"
    tdms_file = os.path.join(base_dir, f"{filename}.tdms")
    parser.parse(
        tdms_file,
        archive,
        logging.getLogger(),
    )
    hdf_file = f"{filename}.hdf"
    assert os.path.exists(hdf_file)
