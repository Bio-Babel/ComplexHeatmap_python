"""
complexheatmap_py — Python port of the R ComplexHeatmap package.
"""

__version__ = "2.25.3+ad11b26"

from complexheatmap._globals import ht_opt, reset_ht_opt
from complexheatmap._color import color_ramp2, add_transparency, rand_color
from complexheatmap._utils import (
    pindex,
    subset_gp,
    max_text_width,
    max_text_height,
    is_abs_unit,
    list_to_matrix,
    restore_matrix,
    default_axis_param,
    cluster_within_group,
    dist2,
)
from complexheatmap.color_mapping import ColorMapping
from complexheatmap.legends import Legend, Legends, pack_legend
from complexheatmap.grid_extensions import (
    grid_boxplot,
    grid_textbox,
    textbox_grob,
    gt_render,
    annotation_axis_grob,
)
from complexheatmap.annotation_function import AnnotationFunction
from complexheatmap.annotation_functions import (
    anno_simple,
    anno_barplot,
    anno_boxplot,
    anno_points,
    anno_lines,
    anno_text,
    anno_histogram,
    anno_density,
    anno_joyplot,
    anno_horizon,
    anno_image,
    anno_link,
    anno_mark,
    anno_block,
    anno_summary,
    anno_empty,
    anno_textbox,
    anno_customize,
    anno_numeric,
    anno_oncoprint_barplot,
)
from complexheatmap.single_annotation import SingleAnnotation
from complexheatmap.heatmap_annotation import (
    HeatmapAnnotation,
    rowAnnotation,
    columnAnnotation,
)
from complexheatmap.heatmap import Heatmap, AdditiveUnit
from complexheatmap.heatmap_list import HeatmapList
from complexheatmap.decorate import (
    decorate_heatmap_body,
    decorate_annotation,
    decorate_column_dend,
    decorate_row_dend,
    decorate_row_names,
    decorate_column_names,
    decorate_row_title,
    decorate_column_title,
    decorate_dimnames,
    list_components,
)
from complexheatmap.upset import (
    make_comb_mat,
    CombMat,
    comb_degree,
    comb_name,
    comb_size,
    set_name,
    set_size,
    extract_comb,
    normalize_comb_mat,
    UpSet,
    upset_top_annotation,
    upset_right_annotation,
    upset_left_annotation,
)
from complexheatmap.oncoprint import (
    oncoPrint,
    alter_graphic,
    test_alter_fun,
)
from complexheatmap.density_heatmap import density_heatmap, frequency_heatmap
from complexheatmap.heatmap_3d import Heatmap3D, bar3D
from complexheatmap._data import (
    load_gene_expression,
    load_measles,
    load_tcga_oncoprint,
    load_sample_order,
    load_dmr_summary,
    load_color_space_comparison,
    load_genome_level_data,
    load_meth_data,
    load_mouse_scrnaseq,
    load_mouse_cell_cycle_genes,
    load_mouse_ribonucleoprotein_genes,
    load_random_meth_expr_data,
)
