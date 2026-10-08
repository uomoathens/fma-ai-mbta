function fma_mbta()
cfg = default_config();
if ~exist(cfg.out, 'dir'), mkdir(cfg.out); end
fprintf('== FMA-AI benchmark on MBTA GTFS ==\n');
fprintf('[1/6] Loading GTFS and building route set... ');
net = build_network(cfg);
write_routes(net, cfg);
fprintf('OK (%d training routes, %d hold-out routes)\n', numel(net.train), numel(net.hold));
methods = {'Centralised', 'Non-Federated MARL', 'FMA-AI (FedAvg)'};
keys = {'central', 'local', 'fedavg'};
nS = numel(cfg.seeds);
rows = {};
curves = zeros(cfg.k_max, numel(keys));
fprintf('[2/6] Training %d seeds x %d methods...\n', nS, numel(keys));
for si = 1:nS
    seed = cfg.seeds(si);
    rng(seed);
    net = with_baseline(net, cfg);
    Zeval = eval_noise(net, cfg);
    for mi = 1:numel(keys)
        [theta, R, hist] = train_method(keys{mi}, net, cfg);
        curves(:, mi) = curves(:, mi) + hist / nS;
        for sp = 1:2
            if sp == 1, idx = net.train; split = 'train'; else, idx = net.hold; split = 'holdout'; end
            S = speeds_for(keys{mi}, theta, net, idx, cfg);
            r = evaluate(S, idx, net, Zeval, cfg);
            rows(end+1, :) = {seed, methods{mi}, split, r.dE, r.otp, r.otp_base, r.enorm, r.enorm_base, r.co2, r.co2_base, R, mean(S)};
        end
    end
    fprintf('  seed %2d done\n', seed);
end
fprintf('[3/6] Writing per-seed results... ');
write_cell_csv(fullfile(cfg.out, 'results_seeds.csv'), ...
    {'seed', 'method', 'split', 'dE_percent', 'OTP_percent', 'OTP_base_percent', 'E_norm_kWh_per_pkm', 'E_norm_base_kWh_per_pkm', 'CO2_t_per_day', 'CO2_base_t_per_day', 'R_conv', 'mean_speed_factor'}, rows);
fprintf('OK\n');
fprintf('[4/6] Summarising... ');
summary = summarise(rows, methods, cfg);
write_cell_csv(fullfile(cfg.out, 'results_summary.csv'), ...
    {'method', 'split', 'n_seeds', 'dE_mean', 'dE_ci95', 'OTP_mean', 'OTP_ci95', 'OTP_base_mean', 'E_norm_mean', 'E_norm_ci95', 'E_norm_base_mean', 'CO2_t_day_mean', 'CO2_base_t_day_mean', 'R_conv_median', 'n_converged'}, summary);
conv_rows = num2cell([(1:cfg.k_max)', curves]);
write_cell_csv(fullfile(cfg.out, 'convergence.csv'), [{'round'}, methods], conv_rows);
fprintf('OK\n');
fprintf('[5/6] OTP tolerance sweep... ');
pareto = tolerance_sweep(net, cfg, methods, keys);
write_cell_csv(fullfile(cfg.out, 'tolerance_sweep.csv'), {'OTP_tolerance_pp', 'method', 'dE_mean', 'dE_ci95', 'OTP_mean', 'OTP_ci95'}, pareto);
fprintf('OK\n');
fprintf('[6/6] Plotting... ');
plot_results(summary, pareto, curves, methods, cfg);
fprintf('OK\nCompleted. Outputs in ./%s\n', cfg.out);
end

function cfg = default_config()
cfg.zip = 'MBTA_GTFS.zip';
cfg.dir = 'mbta_gtfs';
cfg.out = 'results';
cfg.service_date = 20250806;
cfg.target_departure_min = 720;
cfg.n_agents = 6;
cfg.n_holdout = 24;
cfg.pax_avg = 20;
cfg.kwh_per_km = 1.60;
cfg.aux_kw = 3.0;
cfg.cong = 0.15;
cfg.ef = 0.2708;
cfg.delta_max = 2.5;
cfg.bias_sd = 1.0;
cfg.dwell_mu = 1.5;
cfg.dwell_sd = 0.7;
cfg.cong_sd = 1.2;
cfg.s_low = 0.32;
cfg.s_high = 0.10;
cfg.otp_tol = 2;
cfg.rho = 200;
cfg.mu = 0.2;
cfg.tau = 0.5;
cfg.eps = 2e-3;
cfg.b_noise = 64;
cfg.n_base = 4000;
cfg.n_eval = 200;
cfg.k_max = 400;
cfg.e_local = 5;
cfg.eta0 = 0.004;
cfg.kappa = 50;
cfg.delta_conv = 2e-3;
cfg.window = 20;
cfg.seeds = 1:20;
cfg.sweep_tol = [0 1 2 3 5];
cfg.sweep_seeds = 1:5;
end

function net = build_network(cfg)
if ~exist(cfg.dir, 'dir'), unzip(cfg.zip, cfg.dir); end
[hr, rr] = read_csv(fullfile(cfg.dir, 'routes.txt'));
[ht, rt] = read_csv(fullfile(cfg.dir, 'trips.txt'));
[hc, rc] = read_csv(fullfile(cfg.dir, 'calendar.txt'));
[hd, rd] = read_csv(fullfile(cfg.dir, 'calendar_dates.txt'));
active = active_services(hc, rc, hd, rd, cfg.service_date);
route_id = col(hr, rr, 'route_id');
route_type = col(hr, rr, 'route_type');
short_name = col(hr, rr, 'route_short_name');
long_name = col(hr, rr, 'route_long_name');
bus_routes = route_id(strcmp(route_type, '3'));
t_route = col(ht, rt, 'route_id');
t_service = col(ht, rt, 'service_id');
t_trip = col(ht, rt, 'trip_id');
t_dir = col(ht, rt, 'direction_id');
t_shape = col(ht, rt, 'shape_id');
on = ismember(t_service, active) & ismember(t_route, bus_routes);
[ur, ~, ix] = unique(t_route(on));
cnt = accumarray(ix, 1);
[~, order] = sortrows([-cnt, (1:numel(cnt))']);
need = cfg.n_agents + cfg.n_holdout;
cand = ur(order(1:min(numel(order), need + 10)));
cand_cnt = cnt(order(1:numel(cand)));
rep_pool = cell(numel(cand), 1);
all_ids = {};
for c = 1:numel(cand)
    m = on & strcmp(t_route, cand{c}) & strcmp(t_dir, '0');
    sh = t_shape(m);
    ids = t_trip(m);
    if isempty(ids), continue; end
    [us, ~, j] = unique(sh);
    [~, best] = max(accumarray(j, 1));
    rep_pool{c} = struct('ids', {ids(strcmp(sh, us{best}))}, 'shape', us{best});
    all_ids = [all_ids; rep_pool{c}.ids(:)];
end
st = read_stop_times(fullfile(cfg.dir, 'stop_times.txt'), all_ids);
shp = read_shapes(fullfile(cfg.dir, 'shapes.txt'), cellfun(@(p) p.shape, rep_pool(~cellfun('isempty', rep_pool)), 'UniformOutput', false));
k = 0;
net.routes = struct('route_id', {}, 'name', {}, 'n', {}, 'L', {}, 'T', {}, 'tt', {}, 'nstops', {}, 'trip_id', {}, 'shape_id', {});
for c = 1:numel(cand)
    if isempty(rep_pool{c}) || ~isKey(shp, rep_pool{c}.shape), continue; end
    ids = rep_pool{c}.ids;
    have = ids(isKey_all(st, ids));
    if isempty(have), continue; end
    dep = cellfun(@(q) first_el(st(q)), have);
    [~, b] = min(abs(dep - cfg.target_departure_min));
    tt = st(have{b});
    if numel(tt) < 2, continue; end
    k = k + 1;
    ri = find(strcmp(route_id, cand{c}), 1);
    nm = short_name{ri};
    if isempty(nm), nm = long_name{ri}; end
    net.routes(k).route_id = cand{c};
    net.routes(k).name = nm;
    net.routes(k).n = cand_cnt(c);
    net.routes(k).L = shp(rep_pool{c}.shape);
    net.routes(k).T = tt(end) - tt(1);
    net.routes(k).tt = (tt(:) - tt(1))';
    net.routes(k).nstops = numel(tt);
    net.routes(k).trip_id = have{b};
    net.routes(k).shape_id = rep_pool{c}.shape;
    if k == need, break; end
end
net.train = 1:cfg.n_agents;
net.hold = cfg.n_agents + 1:k;
for i = 1:k
    a = net.routes(i);
    net.routes(i).phi = [1; a.L / 20; a.T / 60; a.n / 100; a.nstops / 50];
end
end

function v = first_el(x)
v = x(1);
end

function tf = isKey_all(m, ids)
tf = false(numel(ids), 1);
for i = 1:numel(ids), tf(i) = isKey(m, ids{i}); end
end

function active = active_services(hc, rc, hd, rd, d)
y = floor(d / 10000); mo = mod(floor(d / 100), 100); dd = mod(d, 100);
names = {'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'};
wd = weekday(datenum(y, mo, dd));
svc = col(hc, rc, 'service_id');
flag = str2double(col(hc, rc, names{wd}));
sd = str2double(col(hc, rc, 'start_date'));
ed = str2double(col(hc, rc, 'end_date'));
active = svc(flag == 1 & sd <= d & ed >= d);
dsvc = col(hd, rd, 'service_id');
ddate = str2double(col(hd, rd, 'date'));
dtype = str2double(col(hd, rd, 'exception_type'));
active = union(active, dsvc(ddate == d & dtype == 1));
active = setdiff(active, dsvc(ddate == d & dtype == 2));
end

function st = read_stop_times(fname, ids)
fid = fopen(fname, 'r');
fgetl(fid);
C = textscan(fid, '%s %s %*s %*s %f %*[^\n]', 'Delimiter', ',');
fclose(fid);
keep = ismember(C{1}, ids);
tid = C{1}(keep); tim = C{2}(keep); seq = C{3}(keep);
[~, ~, g] = unique(tid);
[~, o] = sortrows([g, seq]);
tid = tid(o); tim = tim(o); g = g(o);
v = sscanf(strjoin(tim', ' '), '%d:%d:%d');
mins = v(1:3:end) * 60 + v(2:3:end) + v(3:3:end) / 60;
starts = [1; find(diff(g) ~= 0) + 1];
ends = [starts(2:end) - 1; numel(g)];
st = containers.Map();
for q = 1:numel(starts)
    st(tid{starts(q)}) = mins(starts(q):ends(q));
end
end

function shp = read_shapes(fname, ids)
fid = fopen(fname, 'r');
fgetl(fid);
C = textscan(fid, '%s %f %f %f %*[^\n]', 'Delimiter', ',');
fclose(fid);
keep = ismember(C{1}, ids);
sid = C{1}(keep); lat = C{2}(keep); lon = C{3}(keep); seq = C{4}(keep);
[u, ~, g] = unique(sid);
shp = containers.Map();
for q = 1:numel(u)
    m = g == q;
    [~, o] = sort(seq(m));
    la = lat(m); lo = lon(m);
    shp(u{q}) = haversine_path(la(o), lo(o));
end
end

function L = haversine_path(lat, lon)
R = 6371.0088;
lat = lat * pi / 180; lon = lon * pi / 180;
a = sin(diff(lat) / 2) .^ 2 + cos(lat(1:end - 1)) .* cos(lat(2:end)) .* sin(diff(lon) / 2) .^ 2;
L = R * sum(2 * atan2(sqrt(a), sqrt(1 - a)));
end

function [hdr, rows] = read_csv(fname)
txt = fileread(fname);
lines = regexp(txt, '\r?\n', 'split');
lines = lines(~cellfun('isempty', lines));
hdr = regexprep(split_line(lines{1}), '[^A-Za-z0-9_]', '');
n = numel(lines) - 1;
rows = repmat({''}, n, numel(hdr));
for r = 1:n
    f = split_line(lines{r + 1});
    m = min(numel(f), numel(hdr));
    rows(r, 1:m) = f(1:m);
end
end

function f = split_line(s)
if isempty(strfind(s, '"'))
    f = strsplit(s, ',', 'CollapseDelimiters', false);
else
    tok = regexp([s ','], '("(?:[^"]|"")*"|[^,]*),', 'tokens');
    f = cellfun(@(c) strrep(regexprep(c{1}, '^"(.*)"$', '$1'), '""', '"'), tok, 'UniformOutput', false);
end
end

function c = col(hdr, rows, name)
c = rows(:, find(strcmp(hdr, name), 1));
end

function Z = draw_noise(B, m, cfg)
Z = cfg.bias_sd * randn(B, m) + max(0, cfg.dwell_mu + cfg.dwell_sd * randn(B, m)) + cfg.cong_sd * randn(B, m);
end

function Zeval = eval_noise(net, cfg)
Zeval = cell(numel(net.routes), 1);
for i = 1:numel(net.routes)
    Zeval{i} = draw_noise(cfg.n_eval, numel(net.routes(i).tt), cfg);
end
end

function e = energy_trip(s, a, cfg)
e = cfg.kwh_per_km * a.L * (s ^ 2 + cfg.cong * s) + cfg.aux_kw * a.T / (60 * s);
end

function [s, dsdz] = speed(z, cfg)
if z < 0, amp = cfg.s_low; else, amp = cfg.s_high; end
s = 1 + amp * tanh(z);
dsdz = amp * (1 - tanh(z) ^ 2);
end

function o = soft_otp(s, a, Z, cfg)
dt = a.tt * (1 / s - 1) + Z;
o = mean(mean(1 ./ (1 + exp(-(cfg.delta_max - abs(dt)) / cfg.tau))));
end

function net = with_baseline(net, cfg)
for i = 1:numel(net.routes)
    a = net.routes(i);
    net.routes(i).o0 = soft_otp(1, a, draw_noise(cfg.n_base, numel(a.tt), cfg), cfg);
end
end

function l = agent_loss(s, a, Z, cfg)
e = energy_trip(s, a, cfg) / energy_trip(1, a, cfg);
d = max(0, a.o0 - cfg.otp_tol / 100 - soft_otp(s, a, Z, cfg));
l = e + cfg.rho * d ^ 2 + cfg.mu * max(0, s - 1) ^ 2;
end

function g = agent_grad(theta, a, cfg)
[s, dsdz] = speed(theta' * a.phi, cfg);
Z = draw_noise(cfg.b_noise, numel(a.tt), cfg);
d = (agent_loss(s + cfg.eps, a, Z, cfg) - agent_loss(s - cfg.eps, a, Z, cfg)) / (2 * cfg.eps);
g = d * dsdz * a.phi;
end

function [theta, R, hist] = train_method(method, net, cfg)
idx = net.train;
M = numel(idx);
P = numel(net.routes(idx(1)).phi);
n = [net.routes(idx).n];
w = n(:) / sum(n);
if strcmp(method, 'local'), theta = zeros(P, M); nmod = M; else, theta = zeros(P, 1); nmod = 1; end
hist = zeros(cfg.k_max, 1);
R = NaN;
for k = 1:cfg.k_max
    eta = cfg.eta0 / (1 + k / cfg.kappa);
    prev = theta;
    switch method
        case 'central'
            for e = 1:cfg.e_local
                g = zeros(P, 1);
                for i = 1:M
                    g = g + w(i) * agent_grad(theta, net.routes(idx(i)), cfg);
                end
                theta = theta - eta * g;
            end
        case 'fedavg'
            loc = zeros(P, M);
            for i = 1:M
                th = theta;
                for e = 1:cfg.e_local
                    th = th - eta * agent_grad(th, net.routes(idx(i)), cfg);
                end
                loc(:, i) = th;
            end
            theta = loc * w;
        case 'local'
            for i = 1:M
                for e = 1:cfg.e_local
                    theta(:, i) = theta(:, i) - eta * agent_grad(theta(:, i), net.routes(idx(i)), cfg);
                end
            end
    end
    hist(k) = norm(theta(:) - prev(:)) / sqrt(nmod);
    if isnan(R) && k >= cfg.window && mean(hist(k - cfg.window + 1:k)) < cfg.delta_conv, R = k; end
end
end

function S = speeds_for(method, theta, net, idx, cfg)
S = ones(numel(idx), 1);
for j = 1:numel(idx)
    a = net.routes(idx(j));
    if strcmp(method, 'local')
        pos = find(net.train == idx(j), 1);
        if ~isempty(pos), S(j) = speed(theta(:, pos)' * a.phi, cfg); end
    else
        S(j) = speed(theta' * a.phi, cfg);
    end
end
end

function r = evaluate(S, idx, net, Zeval, cfg)
E = 0; Eb = 0; PK = 0; O = 0; Ob = 0; N = 0;
for j = 1:numel(idx)
    a = net.routes(idx(j));
    E = E + a.n * energy_trip(S(j), a, cfg);
    Eb = Eb + a.n * energy_trip(1, a, cfg);
    PK = PK + a.n * cfg.pax_avg * a.L;
    Z = Zeval{idx(j)};
    O = O + a.n * mean(mean(abs(a.tt * (1 / S(j) - 1) + Z) <= cfg.delta_max));
    Ob = Ob + a.n * mean(mean(abs(Z) <= cfg.delta_max));
    N = N + a.n;
end
r.dE = 100 * (1 - E / Eb);
r.otp = 100 * O / N;
r.otp_base = 100 * Ob / N;
r.enorm = E / PK;
r.enorm_base = Eb / PK;
r.co2 = cfg.ef * E / 1000;
r.co2_base = cfg.ef * Eb / 1000;
end

function out = summarise(rows, methods, cfg)
out = {};
splits = {'train', 'holdout'};
for mi = 1:numel(methods)
    for sp = 1:2
        m = strcmp(rows(:, 2), methods{mi}) & strcmp(rows(:, 3), splits{sp});
        if ~any(m), continue; end
        X = cell2mat(rows(m, [4 5 6 7 8 9 10 11]));
        Rv = X(:, 8);
        n = size(X, 1);
        out(end + 1, :) = {methods{mi}, splits{sp}, n, mean(X(:, 1)), ci95(X(:, 1)), mean(X(:, 2)), ci95(X(:, 2)), mean(X(:, 3)), ...
            mean(X(:, 4)), ci95(X(:, 4)), mean(X(:, 5)), mean(X(:, 6)), mean(X(:, 7)), median_or_nan(Rv), sum(~isnan(Rv))};
    end
end
end

function h = ci95(x)
n = numel(x);
if n < 2, h = NaN; return; end
df = n - 1;
q = betaincinv(0.05, df / 2, 0.5);
h = sqrt(df * (1 - q) / q) * std(x) / sqrt(n);
end

function v = median_or_nan(x)
x = x(~isnan(x));
if isempty(x), v = NaN; else, v = median(x); end
end

function out = tolerance_sweep(net, cfg, methods, keys)
out = {};
for wi = 1:numel(cfg.sweep_tol)
    c = cfg;
    c.otp_tol = cfg.sweep_tol(wi);
    for mi = 1:numel(keys)
        dE = zeros(numel(c.sweep_seeds), 1);
        ot = dE;
        for si = 1:numel(c.sweep_seeds)
            rng(1000 + c.sweep_seeds(si));
            net = with_baseline(net, c);
            Zeval = eval_noise(net, c);
            theta = train_method(keys{mi}, net, c);
            r = evaluate(speeds_for(keys{mi}, theta, net, net.train, c), net.train, net, Zeval, c);
            dE(si) = r.dE; ot(si) = r.otp;
        end
        out(end + 1, :) = {c.otp_tol, methods{mi}, mean(dE), ci95(dE), mean(ot), ci95(ot)};
    end
end
end

function write_routes(net, cfg)
rows = cell(numel(net.routes), 9);
for i = 1:numel(net.routes)
    a = net.routes(i);
    if any(net.train == i), role = 'train'; else, role = 'holdout'; end
    rows(i, :) = {role, a.route_id, a.name, a.n, a.L, a.T, a.nstops, a.trip_id, a.shape_id};
end
write_cell_csv(fullfile(cfg.out, 'routes.csv'), {'role', 'route_id', 'route_name', 'trips_per_day', 'length_km', 'runtime_min', 'stops', 'representative_trip_id', 'shape_id'}, rows);
end

function write_cell_csv(fname, hdr, rows)
fid = fopen(fname, 'w');
fprintf(fid, '%s\n', strjoin(hdr, ','));
for r = 1:size(rows, 1)
    parts = cell(1, size(rows, 2));
    for c = 1:size(rows, 2)
        v = rows{r, c};
        if ischar(v)
            parts{c} = ['"' strrep(v, '"', '""') '"'];
        elseif isnan(v)
            parts{c} = 'NaN';
        elseif v == round(v) && abs(v) < 1e9
            parts{c} = sprintf('%d', v);
        else
            parts{c} = sprintf('%.6g', v);
        end
    end
    fprintf(fid, '%s\n', strjoin(parts, ','));
end
fclose(fid);
end

function plot_results(summary, pareto, curves, methods, cfg)
mk = {'s', 'd', 'o'};
f1 = figure('Visible', 'off');
hold on;
tr = strcmp(summary(:, 2), 'train');
S = summary(tr, :);
base_otp = mean(cell2mat(S(:, 8)));
cols = [0.00 0.45 0.74; 0.85 0.33 0.10; 0.47 0.67 0.19];
hp = zeros(1, numel(methods)); he = hp;
for mi = 1:numel(methods)
    p = strcmp(pareto(:, 2), methods{mi});
    hp(mi) = plot(cell2mat(pareto(p, 5)), cell2mat(pareto(p, 3)), ':', 'Color', cols(mi, :), 'LineWidth', 1);
end
for mi = 1:numel(methods)
    q = find(strcmp(S(:, 1), methods{mi}), 1);
    x = S{q, 6}; xe = S{q, 7}; y = S{q, 4}; ye = S{q, 5};
    he(mi) = errorbar(x, y, ye);
    set(he(mi), 'LineStyle', 'none', 'Marker', mk{mi}, 'Color', cols(mi, :), 'MarkerFaceColor', cols(mi, :), 'MarkerSize', 7, 'LineWidth', 1.2);
    plot([x - xe, x + xe], [y, y], '-', 'Color', cols(mi, :), 'LineWidth', 1.2);
end
hb = plot(base_otp, 0, 'kx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('On-time performance (%)');
ylabel('Energy and CO_2 reduction (%)');
legend([hp, he, hb], [strcat(methods, ' (tolerance sweep)'), methods, {'Timetable baseline'}], 'Location', 'northeast');
grid on;
print(f1, fullfile(cfg.out, 'fig_tradeoff.png'), '-dpng', '-r300');
f2 = figure('Visible', 'off');
semilogy(1:cfg.k_max, curves, 'LineWidth', 1.2);
hold on;
semilogy([1 cfg.k_max], [cfg.delta_conv cfg.delta_conv], 'k--');
xlabel('Communication round k');
ylabel('||\theta^{(k)} - \theta^{(k-1)}||_2 (mean over seeds)');
legend([methods, {'\delta'}], 'Location', 'northeast');
grid on;
print(f2, fullfile(cfg.out, 'fig_convergence.png'), '-dpng', '-r300');
end
