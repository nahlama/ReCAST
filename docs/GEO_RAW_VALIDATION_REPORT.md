# GEO cell-level data acquisition and validation

Acquisition date: 2026-07-21

The files listed in `SOURCE_MANIFEST.tsv` were downloaded directly from the
NCBI GEO supplementary-file server. Existing TISCH2 files were not modified.
All six gzip streams were read through end-of-file without a CRC/decompression
error, and SHA-256 checksums were recorded after download.

## GSE160269

GEO describes this as an ESCC single-cell atlas from 60 individuals. The
downloaded matrices are raw UMI counts with genes in rows and cells in columns.

| Compartment | Cells | Genes | Annotation match |
|---|---:|---:|---|
| CD45-negative | 97,631 | 17,012 | Exact cell set and column order |
| CD45-positive | 111,028 | 15,175 | Exact cell set and column order |
| Total | 208,659 | -- | All cell IDs unique across both matrices |

The cell annotation files represent 64 samples: 60 tumour samples and four
adjacent-normal samples. Their coarse labels are:

- CD45-negative: Epithelial 44,730; Fibroblast 37,213; Endothelial 11,267;
  Pericytes 3,102; FRC 1,319.
- CD45-positive: T cell 69,278; B cell 22,477; Myeloid 19,273.

Cross-check against the local TISCH2 metadata:

- all 208,658 TISCH2 cell IDs occur in the downloaded raw-count matrices;
- the raw matrices contain one additional cell,
  `P104T-E-GACAGAGCAAGCTGAG`, which TISCH2 omitted;
- the detailed TISCH2 annotations can therefore be joined to raw counts for
  208,658 cells without ambiguity.

## GSE154763

GEO does not provide raw counts for this series. The public ESCA supplementary
data are nevertheless cell-level, not cell-type summaries:

- 7,673 individual cells;
- 15,550 genes;
- seven patients;
- 4,458 normal-tissue and 3,215 tumour-tissue cells;
- expression rows and metadata rows have identical unique cell IDs in the same
  order.
- all 7,673 GEO cell IDs exactly match the local TISCH2 metadata cell IDs.

The series is restricted to tumour-infiltrating myeloid populations. Its
metadata contains 11 refined monocyte, macrophage, dendritic-cell, pDC, and
mast-cell clusters. It should be used as a specialized myeloid reference, not
as a complete tumour microenvironment reference.

## Use constraints

1. Preserve these compressed source files unchanged.
2. Build sparse derived matrices in a separate processed-data directory.
3. Treat GSE160269 cells as nested within donor/sample during splitting and
   benchmarking; individual cells are not independent biological replicates.
4. Label GSE154763 correctly as normalized cell-level expression, not raw UMI
   counts.
5. Use the local TISCH2 cell metadata as an annotation crosswalk, not as the
   expression source for cell-level modeling.
