if (exist('OCTAVE_VERSION', 'builtin') ~= 0)  % If running Octave. Octave doesn't work yet though.
    pkg load io;
    pkg load mapping;
end

regions_thickness = {'alaska', 'westerncanada', 'arcticcanadaN',...
                     'arcticcanadaS', 'greenland', 'iceland',...
                     'svalbard', 'scandinavia', 'russianarctic',...
                     'northasia', 'centraleurope', 'caucasus',...
                     'centralasiaN', 'centralasiaW', 'centralasiaS',...
                     'lowlatitudes', 'southernandes', 'newzealand',...
                     'antarctic'};

regions_rgi = {'Alaska', 'WesternCanadaUS','ArcticCanadaNorth',...
               'ArcticCanadaSouth', 'GreenlandPeriphery', 'Iceland',...
               'Svalbard', 'Scandinavia', 'RussianArctic', 'NorthAsia',...
               'CentralEurope', 'CaucasusMiddleEast', 'CentralAsia',...
               'SouthAsiaWest', 'SouthAsiaEast', 'LowLatitudes',...
               'SouthernAndes', 'NewZealand', 'AntarcticSubantarctic'};

Regions = struct();

for i = 1:length(regions_rgi)
    if i < 10
        name = strcat('0', num2str(i), '_rgi50_', regions_rgi{i});
    else
        name = strcat(num2str(i), '_rgi50_', regions_rgi{i});
    end
    region = shaperead(strcat(name, '/', name, '.shp'));
    glaciers = zeros(1, length(region));
    for g = 1:length(region)
        if (region(g).GlacType(1) == '0')  % type glacier
            glaciers(g) = str2double(region(g).RGIId(end-4:end));
        else
            glaciers(g) = -1;
        end
    end
    
    glaciers = glaciers(glaciers >= 0);
    
    % Open thickness data
    f = fopen(strcat('thick/thick_', regions_thickness{i}, '_0.00_999.00.dat'));
    d = textscan(f, '%d%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%d%[^\n]',...
                 'Delimiter',{';',' '}, 'MultipleDelimsAsOne',1,...
                 'HeaderLines',1);
    fclose(f);
    
    in_rgi = ismember(d{1,1}, glaciers);   % glaciers in thickness data
                                           % that are present in the RGI
    nonzero_height = d{1,6} > 0;
    nonzero_length = d{1,11} > 0;
    glacier_mask = in_rgi & nonzero_height & nonzero_length;

    Regions.(regions_rgi{i}).('id') = d{1,1}(glacier_mask);
    Regions.(regions_rgi{i}).('heights') = d{1,6}(glacier_mask)/1000;  % mean thickness in km
    Regions.(regions_rgi{i}).('areas') = d{1,4}(glacier_mask);  % areas in km^2
    Regions.(regions_rgi{i}).('volumes') = d{1,5}(glacier_mask);  % volumes in km^3
    Regions.(regions_rgi{i}).('lengths') = d{1,11}(glacier_mask);  % lengths in km
    Regions.(regions_rgi{i}).('widths') = d{1,4}(glacier_mask)./d{1,11}(glacier_mask);  % approximation for widths
    Regions.(regions_rgi{i}).('slopes') = d{1,12}(glacier_mask)*pi/180;  % average slope in radians
end

save('data', 'Regions');