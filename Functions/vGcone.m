function vG = vGcone(vG, origo, width, depth, height, wrist_radius)
    % Width of the structure, height of the structure, thickness of the
    % structure
    % Generates a voxel structure surrounding the arm
    % 1 voxel = 0.1 mm
    radius_x = width/2; % Human arms are not perfectly circular
    radius_y = depth/2; % Human arms are not perfectly circular
    %vG = vGcylinder(vG, origo, 1, radius+thickness, height-1);
    radius = max(radius_x,radius_y);
    diff = radius - wrist_radius;
    height_diffo = height/diff;
    height_diff = height_diffo;
    %vG = vGcylinder(vG, origo, 1, radius+thickness, height-1);
    for z = origo(3) : origo(3) + height
        
        if round(height_diff) < z-origo(3)
            if radius-1 > wrist_radius
                radius = radius - 1;
            end
            height_diff = height_diff + height_diffo;
        end
        r2 = radius * radius;
        for x = origo(1)-radius : origo(1)+radius
           for y = origo(2)-radius : origo(2)+radius
           
               if (x-origo(1))*(x-origo(1)) + (y-origo(2))*(y-origo(2))  < r2 
                   vG(x,y,z) = 1;
               end
           end
       end
    end
    %vG = vGsphere(vG, origo, 1, round(min(radius_x, radius_y)));

    % Generates nodes and edges surrounding the arm
    %{
    h_nodes = 20;
    nh = height/h_nodes;
    w_nodes = 30;
    nw = 360/w_nodes;
    N = [];
    E = [];
    node_nr = 1;
    radius = round((width/2));
    beg = 1;
    for t = 0:thickness/2:thickness
        for i = 0:height_diff:height
            if radius-1 > wrist_radius
                radius = radius - 1;
            end
            for j = 0:nw:360
                x1 = origo(1) + (radius+t)*sind(j);
                y1 = origo(2) + (radius+t)*cosd(j);
                z1 = origo(3) + i;

                N = [N; x1 y1 z1];
                if j == 360
                    E = [E; beg size(N,1)];
                    beg = size(N,1) + 1;
                else
                    E = [E; size(N,1) size(N,1)+1];
                end

                if z1 > origo(3)
                    E = [E; size(N,1)-w_nodes-1 size(N,1)];
                end
            end   
        end
    end
    %}
end
