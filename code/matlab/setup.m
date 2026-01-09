function setup()
%SETUP Add repo paths and load configuration for analysis scripts.

here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

% Load user config if present
if exist(fullfile(here, 'config_paths.m'), 'file')
    cfg = config_paths(); %#ok<NASGU>
    assignin('base', 'cfg', cfg);
else
    warning('No config_paths.m found. Copy config_paths_template.m to config_paths.m and edit paths.');
end
end
