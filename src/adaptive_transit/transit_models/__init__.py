"""Transit-shape models used by injection and detection code."""
from adaptive_transit.transit_models.box import box_transit_mask, box_transit_template
from adaptive_transit.transit_models.periodic import periodic_box_transit_template, transit_center_times

__all__ = ["box_transit_mask", "box_transit_template", "periodic_box_transit_template", "transit_center_times"]
