from torch.utils.data import Dataset
import torch

class UrbanSoundTorchDataset(Dataset):
    def __init__(self, soundata_dataset, fold, transform=None):
        """
        Initialize our urban sound dataset
        """

        # Store arguments
        self.dataset = soundata_dataset
        self.transform = transform
        # handle if fold is just an integer
        if isinstance(fold, int):
            folds = [fold]
        else:
            folds = fold

        # initialise list
        filtered_cid = []
        
        for cid in self.dataset.clip_ids:
            if self.dataset.clip(cid).fold in folds:
                filtered_cid.append(cid)
            else:
                continue
        self.clip_ids = filtered_cid

    def __len__(self):
        """
        Return the length of the filtered clip list
        """
        return len(self.clip_ids)
    
    def __getitem__(self, idx):

        clip_id = self.clip_ids[idx]
        clip = self.dataset.clip(clip_id)
        audio, sr = clip.audio
        label = clip.target

        if self.transform:
            audio = self.transform(audio, sr)
        if not torch.is_tensor(audio):
            audio = torch.tensor(audio, dtype=torch.float32)
        
        return audio, label