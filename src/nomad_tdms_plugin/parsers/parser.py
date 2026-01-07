from typing import (
    TYPE_CHECKING,
)

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
from nptdms import TdmsFile, TdmsWriter, ChannelObject, RootObject, GroupObject

configuration = config.get_plugin_entry_point(
    'nomad_tdms_plugin.parsers:parser_entry_point'
)


class NewParser(MatchingParser):
    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:
        logger.info('NewParser.parse', parameter=configuration.parameter)
        print(TdmsFile.read(mainfile))
        logger.info('NewSchema.parse', f"TdmsFile.read(mainfile)")
        archive.workflow2 = Workflow(name='test')
