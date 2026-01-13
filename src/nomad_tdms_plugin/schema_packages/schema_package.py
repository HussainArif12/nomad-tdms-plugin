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
from nomad.datamodel.data import Schema
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.datamodel.metainfo.plot import PlotSection
from nomad.metainfo import Quantity, SchemaPackage
from nomad.datamodel.hdf5 import HDF5Reference, HDF5Dataset, H5WebAnnotation
from nomad.datamodel import ArchiveSection
import nptdms

configuration = config.get_plugin_entry_point(
    "nomad_tdms_plugin.schema_packages:schema_package_entry_point"
)

m_package = SchemaPackage()


class TDMSDataSchema(Schema, PlotSection, ArchiveSection):

    pass


class NewSchemaPackage(Schema):

    def normalize(self, archive: "EntryArchive", logger: "BoundLogger") -> None:
        super().normalize(archive, logger)

        logger.info("NewSchema.normalize", parameter=configuration.parameter)
        self.message = f"Hello!"


m_package.__init_metainfo__()
