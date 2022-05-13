function comps = check_components(vG)
    comps = bwconncomp(vG, 6).NumObjects;
end