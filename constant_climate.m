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

% The regions and glaciers for the thickness data don't correspond to those
% of RGI5.0, so accumulate all the glaciers from the thickness data into a
% single array for quick access using the RGI IDs.

for i = 1:length(regions_thickness)
    f = fopen(strcat('thick/thick_', regions_thickness{i}, '_0.00_999.00.dat'));
    d = textscan(f, '%d%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%d%[^\n]',...
                 'Delimiter',{';',' '}, 'MultipleDelimsAsOne',1,...
                 'HeaderLines',1);
    fclose(f);
    %                              heights      areas  volumes lengths  widths           slopes
    thickness_data(d{1,1}+1, :) = [d{1, 6}/1000 d{1,4} d{1, 5} d{1, 11} d{1,4}./d{1, 11} d{1,12}*pi/180];
end

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
        glaciers(g) = str2double(region(g).RGIId(end-4:end));
    end
    Regions.(regions_rgi{i}).('id') = glaciers';
    Regions.(regions_rgi{i}).('heights') = thickness_data(glaciers+1, 1);
    Regions.(regions_rgi{i}).('areas') = thickness_data(glaciers+1, 2);
end

AAR = 0.6;
lambda = 1 - exp(-1);

% heights = d{1, 6}/1000;  % mean thickness in km
% mask = find(heights > 0);  % to avoid division by zero
% heights = heights(mask);
% areas = d{1,4}(mask);  % areas in km^2
% volumes = d{1, 5}(mask);  % volumes in km^3
% lengths = d{1, 11}(mask);  % lengths in km
% widths = areas./lengths;  % approximation for widths
% slopes = d{1,12}(mask)*pi/180;  % average slope in radians
Z_ela = heights - AAR*lengths.*tan(slopes);
P = 1 - Z_ela./heights;
final_vols = P*2.*widths.*heights.^2./slopes;
% f = @(AAR) sum(((1 - (heights - AAR*lengths.*tan(slopes))./heights)*2.*widths.*heights.^2./slopes - volumes).^2);
% fminsearch(f, 0.5) %  test: should return around 0.5
timescale = zeros(length(P), 1);
timescale(P>=0) = 1/P(P>=0)*log(1 + lambda/(1 - lambda)*(P(P>=0)./(volumes(P>=0).*(slopes(P>=0)./(2*widths(P>=0).*heights(P>=0).^2)))));