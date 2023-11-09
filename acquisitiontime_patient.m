%% load Dicom
subjectfolder='/Users/mona/Documents/data/registration/patient_noT1/015';


PostconFoldername=[subjectfolder,'/PostconT1'];
listPost=dir(PostconFoldername);
timeino_map = containers.Map();
for n=1:length(listPost)-3
    [PostconFoldername,filesep,listPost(n+3).name]
    if isfolder([PostconFoldername,filesep,listPost(n+3).name])
        DataPost(n)=loaddicom([PostconFoldername,filesep,listPost(n+3).name]);
        timeinstr{n}=DataPost(n).info.AcquisitionTime;
        timeino(n)=str2num(timeinstr{n}(1:2))*60*60+str2num(timeinstr{n}(3:4))*60+str2num(timeinstr{n}(5:end));%s
        timeino_map(listPost(n+3).name) = timeino(n);
    end
end

subject = string(keys(timeino_map));
timeino = cell2mat(values(timeino_map));


t = table(subject', timeino', ...
    'VariableNames',{'Subject', 'AcquisitionTime'});
writetable(t, fullfile(subjectfolder, 'acquisitionTime.csv'));



%%
function data = loaddicom(path)
list=dir(path);
%if ~(strcmp(‘IMA’,list(3).name(end-2:end)))
%  fprintf(‘no Dicom in the directory’)
%else
data.img=dicomread([path,filesep,list(4).name]);
data.info=dicominfo([path,filesep,list(4).name]);
%end
end