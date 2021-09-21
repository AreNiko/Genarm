function vG = vGellipse(vG, origo, add, radiusx, radiusy)
    % Grenser - for loop
    xStart = origo(1) - radiusx - 1;
    yStart = origo(2) - radiusy - 1;
    zStart = origo(3) - radiusx - 1;
    xEnd = origo(1) + radiusx + 1;
    yEnd = origo(2) + radiusy + 1;
    zEnd = origo(3) + radiusx + 1;

    cX = origo(1);
    cY = origo(2);
    cZ = origo(3);

    r2 = radiusx * radiusy; % sparer tid :)
    for x = xStart : xEnd
        for y = yStart : yEnd
            for z = zStart : zEnd
                if (x-cX)*(x-cX) + (y-cY)*(y-cY) + (z-cZ)*(z-cZ) < r2
                    vG(x,y,z) = add;
                end
            end
        end
    end
end