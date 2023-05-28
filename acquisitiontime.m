%% load Dicom
subjectfolder='/Users/mona/Library/CloudStorage/Box-Box/Animals/George/George_wk8+2';
matfilename=dir([subjectfolder,'/DCE/*.mat']);
%if exist([subjectfolder,‘\DCE\workspace.mat’])
if ~isempty(matfilename)
load([subjectfolder,'/DCE/',matfilename(1).name],'Blood', 'HeartMask', 'LVendo', 'Myomask', 'MIMask', 'MVOMask', 'RemoteMask')
% rmdir([subjectfolder,'/DCE/'],'s');
end
% subjectfolder=‘D:\Box\Box\Grants\Cardiac DCE for 7 mins viability scan\Data\code\Data\SofiaD7’;
PreconFoldername=[subjectfolder,'/PreconT1'];
PostconFoldername=[subjectfolder,'/PostconT1'];
listPost=dir(PostconFoldername);
timeino_map = containers.Map();
for n=1:length(listPost)-2
    [PostconFoldername,filesep,listPost(n+2).name]
    if isfolder([PostconFoldername,filesep,listPost(n+2).name])
        DataPost(n)=loaddicom([PostconFoldername,filesep,listPost(n+2).name]);
        timeinstr{n}=DataPost(n).info.AcquisitionTime;
        timeino(n)=str2num(timeinstr{n}(1:2))*60*60+str2num(timeinstr{n}(3:4))*60+str2num(timeinstr{n}(5:end));%s
        timeino_map(listPost(n+2).name) = timeino(n);
    end
end

subject = string(keys(timeino_map));
timeino = cell2mat(values(timeino_map));


t = table(subject', timeino', ...
    'VariableNames',{'Subject', 'AcquisitionTime'});
writetable(t, fullfile(subjectfolder, 'acquisitionTime.csv'));


% listPre=dir(PreconFoldername);
% for n=1:length(listPre)-2
%     DataPre(n)=loaddicom([PreconFoldername,filesep,listPre(n+2).name]);
% end
%%
function data = loaddicom(path)
list=dir(path);
%if ~(strcmp(‘IMA’,list(3).name(end-2:end)))
%  fprintf(‘no Dicom in the directory’)
%else
data.img=dicomread([path,filesep,list(3).name]);
data.info=dicominfo([path,filesep,list(3).name]);
%end
end