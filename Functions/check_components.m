function comps = check_components(vG)
    comps = bwconncomp(vG).NumObjects;
end