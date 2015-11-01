load('data');

regions_rgi = {'Alaska', 'WesternCanadaUS','ArcticCanadaNorth',...
               'ArcticCanadaSouth', 'GreenlandPeriphery', 'Iceland',...
               'Svalbard', 'Scandinavia', 'RussianArctic', 'NorthAsia',...
               'CentralEurope', 'CaucasusMiddleEast', 'CentralAsia',...
               'SouthAsiaWest', 'SouthAsiaEast', 'LowLatitudes',...
               'SouthernAndes', 'NewZealand', 'AntarcticSubantarctic'};

AAR = 0.6;
g = 0.01;
dt = 0.01;
t_f = 500;
lambda = 1 - exp(-1);

for i = 2:length(regions_rgi)
    AAR = 0.6;
    region = Regions.(regions_rgi{i});
    heights = region.heights;
    lengths = region.lengths;
    slopes = region.slopes;
    widths = region.widths;
    Z_ela = heights - AAR*lengths.*tan(slopes);
    volumes = zeros(length(heights), 1);
    volumes(:, 1) = widths.*lengths.*heights;
    for t = 2:t_f/dt
        volumes = volumes + dt*g*((heights - Z_ela).*volumes./heights - (slopes./(2*widths.*heights.^2)).*volumes.^2);
    end
    P = 1 - Z_ela./heights;
    %volumes = widths.*lengths.*heights;
    timescale = (1./P).*log(1 + lambda/(1 - lambda)*(P./(volumes.*(slopes./(2*widths.*heights.^2)))))/g;
    final_vols = volumes;
    %final_vols = P*2.*widths.*heights.^2./slopes;
    
    % perturbation of Z_ela
    Z_ela2 = Z_ela + 0.1;
    P2 = 1 - Z_ela2./heights;
    timescale2 = (1./P2).*log(1 + lambda/(1 - lambda)*(P2./(final_vols.*(slopes./(2*widths.*heights.^2)))))/g;
    volumes = final_vols;
    for t = 2:t_f/dt
        volumes = volumes + dt*g*((heights - Z_ela2).*volumes./heights - (slopes./(2*widths.*heights.^2)).*volumes.^2);
    end
    final_vols2 = volumes;
    %final_vols2 = P2*2.*widths.*heights.^2./slopes;
    disp(strcat(regions_rgi{i}, ', Z_ela perturbation total volume change: ', num2str(sum(final_vols2 - final_vols))));

    % perturbation of AAR
    AAR = 0.7;
    Z_ela3 = heights - AAR*lengths.*tan(slopes);
    P3 = 1 - Z_ela3./heights;
    timescale3 = (1./P3).*log(1 + lambda/(1 - lambda)*(P3./(final_vols.*(slopes./(2*widths.*heights.^2)))))/g;
    volumes = final_vols;
    for t = 2:t_f/dt
        volumes = volumes + dt*g*((heights - Z_ela3).*volumes./heights - (slopes./(2*widths.*heights.^2)).*volumes.^2);
    end
    final_vols3 = volumes;
    %final_vols2 = P2*2.*widths.*heights.^2./slopes;
    disp(strcat(regions_rgi{i}, ', AAR perturbation total volume change: ', num2str(sum(final_vols3 - final_vols))));
end

% f = @(AAR) sum(((1 - (heights - AAR*lengths.*tan(slopes))./heights)*2.*widths.*heights.^2./slopes - volumes).^2);
% fminsearch(f, 0.5) %  test: should return around 0.5