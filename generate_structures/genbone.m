function [vG, figcount] = genbone(env, origo, b_layer, f_layer, max_radius, wrist_radius, height, thickness, gt, weight_hand, weight_want, figcount)
    vG_work = vGcylinder(env,[origo(1),origo(2),b_layer+max_radius+1],1, thickness-1,f_layer-(b_layer+ max_radius-thickness+1)); % Middel tube/Bone 
    vG_work1 = vG_work;
    
    start_coord = [];
    end_coord = [];
    directprob = [];
    x = [];
    y = [];
    z = [];
    don = [];
    radi = floor(thickness/2)-1;
    beam = 20;
    
    height_beam = height/beam;
    for i = 0:beam-2
        
        x0 = origo(1);
        y0 = origo(2);
        %z0 = randi([b_layer+max_radius+1,f_layer]);
        z0 = origo(3) + max_radius + thickness + round(height_beam*i); 
        %fprintf('%.5f %.5f \n', cosd(theta), sind(theta));
        %fprintf('%d %d %d \n', theta, x0, y0); 
        x = [x x0];
        y = [y y0];
        z = [z z0];
        start_coord = [start_coord; x0 y0 z0];
        don = [don 0];
        directprob = [directprob; 0.2 0.4 0.6 0.8];
        
    end
    %x = origo(1)-(max_radius+1)*cosd(0); y = origo(2)-(max_radius+1)*sind(0); z = b_layer+2;
    origo_z = b_layer+1;
    
    amount_t1 = sum(vG_work1,'all');
    
    figcount=figcount+1;figure(figcount);clf;plotVg_safe(vG_work1,'edgeOff');
    sun1;
    
    save_locked = env;
    save_force = env;
    
    diff = max_radius - wrist_radius;
    height_diffo = height/diff;
    
    step_count = 0;
    coord_list = [];
    gener = 1;
    %gt = 1000;
    
    disp([length(don) sum(don)]);
    
    direc = 1;
    for i=3:beam-3
        level = floor((z(i)-origo(3))/height_diffo);
        direc = direc + 1;
        for j = 1:(max_radius-level)
            if mod(direc,4) == 0
                x(i) = x(i) + 1;
            elseif mod(direc,4) == 1
                y(i) = y(i) + 1;
            elseif mod(direc,4) == 2
                x(i) = x(i) - 1;
            else
                y(i) = y(i) - 1;
            end
            vG_work1(x(i)-radi:x(i)+radi,y(i)-radi:y(i)+radi,z(i)-radi:z(i)+radi) = 1;
        end
    end
    vG = vG_work1;
    %{
    %while amount_t1 < wanted_nrnodes
    while length(don) > sum(don)
        fprintf("Gen %d\n", gener);
        gener = gener + 1;
        step_count = step_count + 1;
        amount_t0 = amount_t1;
        
        disp([length(don) sum(don)]);
        for i = 1:length(x)
            level = floor((z(i)-origo(3))/height_diffo);
            angdist = max_radius-level;
            fprintf('Wall: %.5f | Current: %.5f \n', angdist, sqrt((x(i)-origo(1))*(x(i)-origo(1))+(y(i)-origo(2))*(y(i)-origo(2))))
            
            if don(i) == 1
                check = 1;
            else
                check = 0;
            end
            
            while check == 0
                direct = rand;
                %fprintf('%d ' , direct);
                
                %if direct < 0.22 % Left
                if direct < directprob(i,1) % Left
                    if sqrt((x(i)-1-origo(1))*(x(i)-1-origo(1))+(y(i)-origo(2))*(y(i)-origo(2))) < angdist
                    %if x - 1 > origo(1)-(max_radius+5)
                        x(i) = x(i) - 1;
                        check = 1;
                        
                        
                        directprob(i,1) = directprob(i,1)+0.005;
                        
                    else
                        directprob(i,1) = directprob(i,1)-0.01;
                        don(i) = 1;
                        
                    end
                    
                %elseif 0.22 < direct && direct < 0.44 % Forward
                elseif directprob(i,1) < direct && direct < directprob(i,2) % Forward
                    if sqrt((x(i)-origo(1))*(x(i)-origo(1))+((y(i)+1-origo(2))*(y(i)+1-origo(2)))) < angdist
                    %if y + 1 < origo(2)+(max_radius+5)
                        y(i) = y(i) + 1;
                        check = 1;
                        
                        if directprob(i,2)-directprob(i,1) < directprob(i,3)-directprob(i,2)
                            directprob(i,2) = directprob(i,2)+0.005;
                        else
                            directprob(i,1) = directprob(i,1)-0.005;
                        
                        end
                        
                    else
                        directprob(i,1) = directprob(i,1)+0.005;
                        directprob(i,2) = directprob(i,2)-0.005;
                        don(i) = 1;
                    end
                    
                %elseif 0.44 < direct && direct < 0.66 % Right
                elseif directprob(i,2) < direct && direct < directprob(i,3) % Right
                    if sqrt(((x(i)+1-origo(1))*(x(i)+1-origo(1)))+((y(i)-origo(2))*(y(i)-origo(2)))) < angdist
                    %if x + 1 < origo(1)+(max_radius+5)
                        x(i) = x(i) + 1;
                        check = 1;
                        
                        if directprob(i,3)-directprob(i,2) < directprob(i,4)-directprob(i,3)
                            directprob(i,3) = directprob(i,3)+0.005;
                        else
                            directprob(i,2) = directprob(i,2)-0.005;
                        end
                        
                    else
                        directprob(i,2) = directprob(i,2)+0.005;
                        directprob(i,3) = directprob(i,3)-0.005;
                        don(i) = 1;
                    end
                %elseif 0.66 < direct && direct < 0.88  % Back
                elseif directprob(i,3) < direct && direct < directprob(i,4) % Back
                    if sqrt(((x(i)-origo(1))*(x(i)-origo(1)))+((y(i)-1-origo(2))*(y(i)-1-origo(2)))) < angdist
                    %if y - 1 > origo(2)-(max_radius+5)
                        y(i) = y(i) - 1;
                        check = 1;
                        
                        if directprob(i,3)-directprob(i,2) < directprob(i,4)-directprob(i,3)
                            directprob(i,3) = directprob(i,3)+0.005;
                        else
                            directprob(i,2) = directprob(i,2)-0.005;
                        end
                        
                    else
                        directprob(i,2) = directprob(i,2)+0.005;
                        directprob(i,3) = directprob(i,3)-0.005;
                        don(i) = 1;
                    end
 
                else % Up
                    updown = rand;
                    if updown < 0.5
                        if z(i)-1 > b_layer+max_radius+1
                            z(i) = z(i) - 1;
                            check = 1;
                        end
                    else
                        if z(i)+1 < f_layer+thickness-1
                            z(i) = z(i) + 1;
                            check = 1;
                        end
                    end
                end
            end
            %fprintf('(%d, %d, %d)\n', x, y, z);
            %coord_list = [coord_list; x y z];
            
            vG_work1(x(i)-radi:x(i)+radi,y(i)-radi:y(i)+radi,z(i)-radi:z(i)+radi) = 1; 
            
            %{
            new_branch = rand;
            if new_branch < 0.01 && z(i) > origo_z

                save_locked(x(i),y(i),z(i)+radi) = vG_work(x(i),y(i),z(i)+radi);
                save_force(x(i),y(i),z(i)+radi-1) = vG_work(x(i),y(i),z(i)+radi-1);
                previdx = randi(length(coord_list));
                prev_line = coord_list(previdx, :);
                %coord_list = coord_list(1:previdx, :);
                x = prev_line(1); y = prev_line(2); z = prev_line(3);
            end
            %}
        end
        
        figure(figcount);clf;plotVg_safe(vG_work1,'edgeOff');
        title("(Stayoff (Gul), extF (grå), låst (blå)");
        sun1;

        saveFigToAnimGif('bone_growing.gif', step_count==1);

        amount_t1 = sum(vG_work1,'all');
        fprintf('Amount of nodes: %d', amount_t1);
        fprintf("\n");
        
    end
    %}
    %{
    for i = 1:length(x)
        x1 = x(i) - start_coord(i, 1);
        y1 = y(i) - start_coord(i, 2);
        z1 = z(i) - start_coord(i, 3);
        
        m1 = y1/x1;
        xs = 0;
        while xs ~= x(i) && y1 ~= y(i)
            y1 = m1*xs;
            vG_work(origo(1)+xs-radi:origo(1)+xs+radi,origo(2)+y1-radi:origo(2)+y1+radi,start_coord(i, 3)-radi:start_coord(i, 3)+radi) = 1; 
            xs = xs + 1;
            
            figure(figcount);clf;plotVg_safe(vG_work,'edgeOff');
            title("(Stayoff (Gul), extF (grå), låst (blå)");
            sun1;
        end
    end
    %}
    
end