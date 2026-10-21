"""Rasterio-backed metadata inspection and lazy, grid-aware window reading."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from typing_extensions import Self

from .contracts import AlignmentError, GridSpec, RasterAsset, SceneRecord, WindowSpec


def _rasterio() -> Any:
    try:
        import rasterio
    except ImportError as error:
        raise RuntimeError(
            "Rasterio is required for geospatial data access. Install the project with "
            "`pip install -e .`."
        ) from error
    return rasterio


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file hash during ingest; never hash per training step."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_raster(
    path: Path,
    *,
    asset_id: str,
    uri: str,
    bands: tuple[str, ...],
    grid_id: str,
    alignment_evidence: str | None = None,
    hash_file: bool = False,
) -> RasterAsset:
    """Inspect raster dimensions, dtype, grid, nodata, and optional content hash."""
    rasterio = _rasterio()
    path = Path(path)
    with rasterio.open(path) as source:
        if source.count != len(bands):
            raise ValueError(
                f"{path} has {source.count} bands but the catalog declares {len(bands)}"
            )
        if len(set(source.dtypes)) != 1:
            raise ValueError(
                f"Mixed per-band dtypes are not supported for {path}: {source.dtypes}"
            )
        crs = source.crs.to_string() if source.crs else None
        transform = tuple(float(value) for value in tuple(source.transform)[:6])
        identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        has_transform = not all(
            abs(value - identity_value) <= 1e-15
            for value, identity_value in zip(transform, identity, strict=True)
        )
        affine = transform if crs is not None or has_transform else None
        grid = GridSpec(
            grid_id=grid_id,
            width=source.width,
            height=source.height,
            crs=crs,
            affine=affine,
        )
        nodata = source.nodata
        if isinstance(nodata, float) and not np.isfinite(nodata):
            nodata = str(nodata)
        dtype = source.dtypes[0]

    return RasterAsset(
        asset_id=asset_id,
        uri=uri,
        bands=bands,
        dtype=dtype,
        grid=grid,
        nodata=nodata,
        sha256=sha256_file(path) if hash_file else None,
        alignment_evidence=alignment_evidence,
    )


def verify_raster_metadata(path: Path, asset: RasterAsset) -> None:
    """Fail if the on-disk raster no longer matches its catalog record."""
    rasterio = _rasterio()
    with rasterio.open(path) as source:
        if source.count != len(asset.bands):
            raise ValueError(f"Band count changed for {asset.uri}")
        if len(set(source.dtypes)) != 1 or source.dtypes[0] != asset.dtype:
            raise ValueError(f"Raster dtype changed for {asset.uri}")
        if (source.width, source.height) != (asset.grid.width, asset.grid.height):
            raise ValueError(f"Raster dimensions changed for {asset.uri}")
        actual_crs = source.crs.to_string() if source.crs else None
        actual_affine = tuple(float(value) for value in tuple(source.transform)[:6])
        identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        has_transform = not all(
            abs(value - identity_value) <= 1e-15
            for value, identity_value in zip(actual_affine, identity, strict=True)
        )
        if actual_crs is None and not has_transform:
            actual_affine = None
        expected_grid = GridSpec(
            grid_id=asset.grid.grid_id,
            width=source.width,
            height=source.height,
            crs=actual_crs,
            affine=actual_affine,
        )
        if not expected_grid.same_grid(asset.grid):
            raise AlignmentError(
                f"CRS/affine changed for {asset.uri}: expected "
                f"{asset.grid}, got {expected_grid}"
            )
        if asset.nodata is not None and str(source.nodata) != str(asset.nodata):
            raise ValueError(f"Nodata metadata changed for {asset.uri}")


class RasterioSceneReader:
    """Read lazy windows from a scene and enforce the catalog's alignment evidence."""

    def __init__(
        self,
        scene: SceneRecord,
        data_root: Path,
        *,
        allow_reprojection: bool = False,
        verify_metadata: bool = True,
    ) -> None:
        self.scene = scene
        self.data_root = Path(data_root).expanduser().resolve()
        self.allow_reprojection = allow_reprojection
        self.verify_metadata = verify_metadata
        self._datasets: dict[str, Any] = {}

    def _path(self, asset: RasterAsset) -> Path:
        relative = PurePosixPath(asset.uri)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe asset URI: {asset.uri}")
        path = (self.data_root / Path(*relative.parts)).resolve()
        if path != self.data_root and self.data_root not in path.parents:
            raise ValueError(f"Asset URI escapes the configured data root: {asset.uri}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _open(self, asset_name: str) -> Any:
        assets = self.scene.asset_map()
        if asset_name not in assets:
            raise KeyError(f"Scene {self.scene.scene_id} has no asset {asset_name!r}")
        if asset_name not in self._datasets:
            rasterio = _rasterio()
            asset = assets[asset_name]
            path = self._path(asset)
            if self.verify_metadata:
                verify_raster_metadata(path, asset)
            self._datasets[asset_name] = rasterio.open(path)
        return self._datasets[asset_name]

    def read_window(
        self,
        asset_name: str,
        window: WindowSpec,
        *,
        resampling: str = "nearest",
        fill_value: float = 0,
    ) -> np.ndarray:
        """Return HWC pixels, reprojecting only between georeferenced grids."""
        _rasterio()
        from affine import Affine
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT
        from rasterio.windows import Window

        assets = self.scene.asset_map()
        asset = assets[asset_name]
        window.validate_for(self.scene.reference_grid)
        self.scene.require_aligned(
            (asset_name,), allow_reprojection=self.allow_reprojection
        )
        source = self._open(asset_name)
        attested_pixel_pair = (
            (asset.grid.width, asset.grid.height)
            == (self.scene.reference_grid.width, self.scene.reference_grid.height)
            and asset.alignment_evidence is not None
            and (
                not asset.grid.georeferenced
                or not self.scene.reference_grid.georeferenced
            )
        )
        source_grid_matches = (
            asset.grid.same_grid(self.scene.reference_grid) or attested_pixel_pair
        )
        if source_grid_matches:
            result = source.read(
                window=Window(window.col, window.row, window.width, window.height),
                boundless=window.padding == "constant",
                fill_value=fill_value,
            )
        elif self.allow_reprojection and asset.grid.georeferenced:
            if not self.scene.reference_grid.georeferenced:
                raise AlignmentError(
                    "Cannot reproject into a pixel-local reference grid"
                )
            try:
                resampling_method = getattr(Resampling, resampling)
            except AttributeError as error:
                raise ValueError(
                    f"Unknown raster resampling method: {resampling}"
                ) from error
            reference = self.scene.reference_grid
            assert reference.affine is not None and reference.crs is not None
            with WarpedVRT(
                source,
                crs=reference.crs,
                transform=Affine(*reference.affine),
                width=reference.width,
                height=reference.height,
                resampling=resampling_method,
            ) as warped:
                result = warped.read(
                    window=Window(window.col, window.row, window.width, window.height),
                    boundless=window.padding == "constant",
                    fill_value=fill_value,
                )
        else:
            raise AlignmentError(
                f"Asset {asset_name!r} does not match the scene reference grid"
            )
        return np.ascontiguousarray(result.transpose(1, 2, 0))

    def read_window_with_mask(
        self,
        asset_name: str,
        window: WindowSpec,
        *,
        resampling: str = "nearest",
        fill_value: float = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return HWC pixels and a 2D mask separate from categorical labels."""
        _rasterio()
        from affine import Affine
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT
        from rasterio.windows import Window

        assets = self.scene.asset_map()
        asset = assets[asset_name]
        window.validate_for(self.scene.reference_grid)
        self.scene.require_aligned(
            (asset_name,), allow_reprojection=self.allow_reprojection
        )
        source = self._open(asset_name)
        window_arg = Window(window.col, window.row, window.width, window.height)
        same_grid = asset.grid.same_grid(self.scene.reference_grid)
        attested_pair = (
            (asset.grid.width, asset.grid.height)
            == (self.scene.reference_grid.width, self.scene.reference_grid.height)
            and asset.alignment_evidence is not None
            and (
                not asset.grid.georeferenced
                or not self.scene.reference_grid.georeferenced
            )
        )
        if same_grid or attested_pair:
            data = source.read(
                window=window_arg,
                boundless=window.padding == "constant",
                fill_value=fill_value,
            )
            masks = source.read_masks(
                window=window_arg,
                boundless=window.padding == "constant",
            )
        elif self.allow_reprojection and asset.grid.georeferenced:
            reference = self.scene.reference_grid
            if not reference.georeferenced:
                raise AlignmentError(
                    "Cannot reproject into a pixel-local reference grid"
                )
            try:
                resampling_method = getattr(Resampling, resampling)
            except AttributeError as error:
                raise ValueError(
                    f"Unknown raster resampling method: {resampling}"
                ) from error
            assert reference.affine is not None and reference.crs is not None
            with WarpedVRT(
                source,
                crs=reference.crs,
                transform=Affine(*reference.affine),
                width=reference.width,
                height=reference.height,
                resampling=resampling_method,
            ) as warped:
                data = warped.read(
                    window=window_arg,
                    boundless=window.padding == "constant",
                    fill_value=fill_value,
                )
                masks = warped.read_masks(
                    window=window_arg,
                    boundless=window.padding == "constant",
                )
        else:
            raise AlignmentError(
                f"Asset {asset_name!r} does not match the scene reference grid"
            )
        valid = np.all(masks > 0, axis=0)
        return np.ascontiguousarray(data.transpose(1, 2, 0)), valid

    def close(self) -> None:
        for dataset in self._datasets.values():
            dataset.close()
        self._datasets.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        """Do not pickle open GDAL handles into DataLoader worker processes."""
        state = self.__dict__.copy()
        state["_datasets"] = {}
        return state
