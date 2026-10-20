% Download Big 12 conference games for 2025 and 2026 from ESPN's public
% scoreboard and write big12.csv. Run from hw6/; rerun after each weekend.
% Unplayed games are kept (scores NaN) so they can be predicted.

url = ['https://site.api.espn.com/apis/site/v2/sports/football/college-football/' ...
       'scoreboard?seasontype=2&groups=4&limit=200'];
opts = weboptions('Timeout',30);
T = table('Size',[0 9],'VariableTypes',["double","double","string","string","double", ...
          "string","double","double","double"],'VariableNames',["season","week", ...
          "date","home","home_pts","away","away_pts","neutral","completed"]);
for season = [2025 2026]
    for w = 1:15
        d = webread(sprintf('%s&dates=%d&week=%d',url,season,w),opts);
        if ~isfield(d,'events') || isempty(d.events), continue, end
        ev = d.events; if isstruct(ev), ev = num2cell(ev); end
        for k = 1:numel(ev)
            c = ev{k}.competitions; if iscell(c), c = c{1}; end; c = c(1);
            cp = c.competitors; if isstruct(cp), cp = num2cell(cp); end
            conf = cellfun(@(x) string(getfield_or(x.team,'conferenceId','')), cp);
            if ~all(conf == "4"), continue, end                % Big 12 vs Big 12
            ha = cellfun(@(x) string(x.homeAway), cp); H = cp{ha=="home"}; A = cp{ha=="away"};
            done = c.status.type.completed;
            sc = @(x) str2double(x.score) + 0/double(done);    % NaN until played
            T(end+1,:) = {season, w, string(ev{k}.date(1:10)), string(H.team.location), ...
                sc(H), string(A.team.location), sc(A), double(c.neutralSite), double(done)}; %#ok<SAGROW>
        end
    end
end
T = unique(T,'stable');
writetable(T,'big12.csv');
fprintf('%d games written to big12.csv (2025: %d, 2026 played: %d, 2026 to play: %d)\n', ...
    height(T), sum(T.season == 2025), sum(T.season == 2026 & T.completed == 1), ...
    sum(T.season == 2026 & T.completed == 0));

function v = getfield_or(s,f,default)
if isfield(s,f), v = s.(f); else, v = default; end
end
