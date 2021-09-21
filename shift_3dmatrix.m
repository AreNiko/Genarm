function nvG = shift_3dmatrix(vG, x_shift, y_shift, z_shift)
    [xlen, ylen, zlen] = size(vG);
    nvGx = zeros(size(vG),'int8');
    for x = 1:xlen
        if sum(vG(x,:,:), 'all') > 0
            if x+x_shift > xlen && x+x_shift < 2
                error('Out of bound');
            end
            nvGx(x+x_shift,:,:) = vG(x,:,:);
        end
    end
    if y_shift ~= 0
        nvGy = zeros(size(vG),'int8');
        for y = 1:ylen
            if sum(vG(:,y, :), 'all') > 0
                if y+y_shift > ylen && y+y_shift < 2
                    error('Out of bound');
                end
                nvGy(:,y+y_shift,:) = nvGx(:,y,:);
            end
        end
    else
        nvGy = nvGx;
    end
    
    
    
    if z_shift ~= 0
        nvGz = zeros(size(vG),'int8');
        for z = 1:zlen
            if sum(vG(:,:,z), 'all') > 0
                if z+z_shift > zlen && z+z_shift < 2
                    error('Out of bound');
                end
                nvGz(:,:,z+z_shift) = nvGy(:,:,z);
            end
        end
    else
        nvGz = nvGy;
    end
    
    nvG = nvGz;
end