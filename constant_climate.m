load('data');

AAR = 0.6;
lambda = 1 - exp(-1);

Z_ela = heights - AAR*lengths.*tan(slopes);
P = 1 - Z_ela./heights;
final_vols = P*2.*widths.*heights.^2./slopes;
% f = @(AAR) sum(((1 - (heights - AAR*lengths.*tan(slopes))./heights)*2.*widths.*heights.^2./slopes - volumes).^2);
% fminsearch(f, 0.5) %  test: should return around 0.5
timescale = zeros(length(P), 1);
timescale(P>=0) = 1/P(P>=0)*log(1 + lambda/(1 - lambda)*(P(P>=0)./(volumes(P>=0).*(slopes(P>=0)./(2*widths(P>=0).*heights(P>=0).^2)))));