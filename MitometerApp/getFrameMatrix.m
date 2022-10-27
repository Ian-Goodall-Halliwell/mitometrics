function frameMatrix = getFrameMatrix(mito,track)

numMito = length(mito);
numTracks = length(track);

%initialize
frameMatrix = zeros(numMito,numTracks);
frm = mito.frame;
disp (frm);
frameMito = [mito.frame];

for trackNum = 1:numTracks
    frameMatrix(:,trackNum) = frameMito'-track(trackNum).frame(end);
end

end