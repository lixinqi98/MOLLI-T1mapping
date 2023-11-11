%% load Dicom
subjectfolder='/Users/mona/Library/CloudStorage/GoogleDrive-xinqili16@g.ucla.edu/My Drive/Registration/patient_Liting_dDCE_Mona';


PostconFoldername=[subjectfolder,'/075'];
listPost=dir(sprintf('%s/PostconT1/*_MAP_T1_Mona', PostconFoldername));
timeino_map = containers.Map();
for n=1:length(listPost)
    file = fullfile(listPost(n).folder, listPost(n).name)
    if isfolder(file)
        DataPost(n)=loaddicom(file);
        timeinstr{n}=DataPost(n).info.AcquisitionTime;
        timeino(n)=str2num(timeinstr{n}(1:2))*60*60+str2num(timeinstr{n}(3:4))*60+str2num(timeinstr{n}(5:end));%s
        timeino_map(listPost(n).name) = timeino(n);
    end
end

subject = string(keys(timeino_map));
timeino = cell2mat(values(timeino_map));


t = table(subject', timeino', ...
    'VariableNames',{'Subject', 'AcquisitionTime'});
writetable(t, fullfile(PostconFoldername, 'acquisitionTime.csv'));



%%
function data = loaddicom(path)
list=dir(sprintf('%s/*.dcm', path));
%if ~(strcmp('IMA',list(3).name(end-2:end)))

%  fprintf('no Dicom in the directory')  
%else
% Mona: change the 3 to 'end', due to exist of .DS_Store
data.img=dicomread([path,filesep,list(end).name]);
data.info=dicominfo([path,filesep,list(end).name]);

end