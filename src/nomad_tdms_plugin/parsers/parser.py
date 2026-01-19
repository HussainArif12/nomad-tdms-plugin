import os
import time
from typing import (
    TYPE_CHECKING,
)

from nomad_tdms_plugin.schema_packages.schema_package import NewSchemaPackage

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )

from nomad.config import config
from nomad.datamodel.metainfo.workflow import Workflow
from nomad.parsing.parser import MatchingParser
from nptdms import TdmsFile
from nomad.files import StagingUploadFiles
from .parse_tdms_helpers import (
    INDEX_MAPPING,
    detect_cycles,
    extract_index_series_all,
    get_file_timerange,
    load_tdms_file,
    save_cycle,
    timed,
    filter_cycle,
)

configuration = config.get_plugin_entry_point(
    "nomad_tdms_plugin.parsers:parser_entry_point"
)

_parse_call_count = {}


class NewParser(MatchingParser):
    def parse(
        self,
        mainfile: str,
        archive: "EntryArchive",
        logger: "BoundLogger",
        child_archives: dict[str, "EntryArchive"] = None,
    ) -> None:
        logger.info("NewParser.parse", parameter="File running")

        print(f"🔍 PARSE CALL #{call_num} - File: {mainfile}, Upload: {upload_id}")
        print(mainfile)

        # upload_id = archive.m_context.upload_id
        # upload_id_first_chars = upload_id[:2]

        archive.metadata.upload_id = archive.m_context.upload_id  # upload_id
        archive.metadata.entry_id = "tdms_dataset"
        archive.data = NewSchemaPackage()
        upload_id = archive.m_context.upload_id if archive.m_context else "tdms_upload"
        StagingUploadFiles(
            upload_id=upload_id,
            create=True,
        )

        upload_id = archive.m_context.upload_id if archive.m_context else "unknown"
        key = f"{upload_id}:{mainfile}"
        _parse_call_count[key] = _parse_call_count.get(key, 0) + 1
        call_num = _parse_call_count[key]

        logger.info(
            "NewParser.parse",
            call_number=call_num,
            mainfile=mainfile,
            upload_id=upload_id,
        )
        tdms_file_paths = load_tdms_file(mainfile)
        logger.info("NewSchema.parse", parameter=f"{load_tdms_file(mainfile)}")

        file_ranges = {f: get_file_timerange(f) for f in tdms_file_paths}
        index_df = timed(
            "Index laden", extract_index_series_all, logger, tdms_file_paths
        )
        cycles = timed("Zykluserkennung", detect_cycles, logger, index_df)

        zyklus_counter = {}
        t_start = time.perf_counter()

        for i, (c, unvollst) in enumerate(cycles, start=1):
            print(f"\n▶ Zyklus {i}/{len(cycles)} – Start")

            # Mapping bestimmen
            indices = set(c["index"].values)
            marker = [idx for idx in indices if idx in INDEX_MAPPING]
            if not marker:
                logger.info(
                    "NewSchema.parse",
                    parameter=f" ⚠ Kein Mapping (TYP/TEMP/Zustand) gefunden, Zyklus übersprungen",
                )
                print(
                    "   ⚠ Kein Mapping (TYP/TEMP/Zustand) gefunden, Zyklus übersprungen"
                )
                continue
            typ, temp, zustand = INDEX_MAPPING[marker[0]]

            key = (typ, temp, zustand)
            zyklus_counter[key] = zyklus_counter.get(key, 0) + 1

            # Filtern & Speichern
            cycle_data = timed(
                f"Filter Zyklus {i}",
                filter_cycle,
                logger,
                tdms_file_paths,
                c,
                file_ranges,
            )
            timed(
                f"Speichern Zyklus {i}",
                save_cycle,
                logger,
                archive.m_context,
                archive,
                cycle_data,
                typ,
                temp,
                zustand,
                zyklus_counter[key],
                c,
                unvollst,
            )

            elapsed = time.perf_counter() - t_start
            logger.info(
                "NewParser.parse", f"   ✔ Zyklus {i} fertig | bisher {elapsed:.1f}s"
            )

        archive.workflow2 = Workflow(name="test")
