"""Download and preprocess data into several random train/test splits."""
import numpy as np
from thu_rsvp_dataset import THU_RSVP_Dataset, get_default_transform

from bci_disc_models.utils import PROJECT_ROOT, seed_everything

# Load all trials data from THU dataset
transform = get_default_transform(
    sample_rate_hz=250,  # Sample rate of original dataset
    notch_freq_hz=50,  # AC line frequency in China
    notch_quality_factor=30,
    bandpass_low=1,  # Low frequency cutoff
    bandpass_high=20,  # High frequency cutoff
    bandpass_order=2,  # Order of Butterworth filter
    downsample_factor=2,  # Downsample by 2
)
dataset = THU_RSVP_Dataset(
    dir=PROJECT_ROOT / "datasets",
    trial_duration_ms=500,
    transform=transform,
    download=True,
    verify_sha256=False,
    verbose=True,
    force_extract=False,  # NOTE - set this to true after changing transforms
)
all_data, all_labels, all_subj_id, all_sess_id = dataset.get_data()

print(np.unique(all_labels))
print(np.unique(all_subj_id))
print(np.unique(all_sess_id))

# Save a little memory. Later, convert back to 'int' dtypes as needed
print(all_data.dtype, all_labels.dtype, all_subj_id.dtype, all_sess_id.dtype)
all_labels = all_labels.astype(bool)
all_subj_id = all_subj_id.astype(np.uint8)
all_sess_id = all_sess_id.astype(bool)
print(all_data.dtype, all_labels.dtype, all_subj_id.dtype, all_sess_id.dtype)


def shuffle_together(seed, arrays):
    seed_everything(seed)
    first_len = len(arrays[0])
    assert all(len(x) == first_len for x in arrays)

    perm = np.random.permutation(first_len)
    return [x[perm] for x in arrays]


a, b = shuffle_together(0, [np.arange(15), np.arange(15)])
print(a)
print(b)


def data_one_split(seed):
    train_x, train_y, test_x, test_y = [], [], [], []
    train_fraction = 0.8
    for subj in np.unique(all_subj_id):
        subj_idx = all_subj_id == subj

        subj_x = all_data[subj_idx]
        subj_y = all_labels[subj_idx]
        subj_sess = all_sess_id[subj_idx]

        for sess in np.unique(subj_sess):
            sess_idx = subj_sess == sess

            sess_x = subj_x[sess_idx]
            sess_y = subj_y[sess_idx]
            sess_x, sess_y = shuffle_together(seed, [sess_x, sess_y])

            n_train = int(train_fraction * len(sess_x))
            train_x.append(sess_x[:n_train])
            train_y.append(sess_y[:n_train])
            test_x.append(sess_x[n_train:])
            test_y.append(sess_y[n_train:])

    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    test_x = np.concatenate(test_x)
    test_y = np.concatenate(test_y)

    return train_x, train_y, test_x, test_y


# Split 80/20 for each session, then combine all the 80%s to become train, and the 20%s to become test
# Save several splits for cross-validation

for seed in range(5):
    train_x, train_y, test_x, test_y = data_one_split(seed)
    print("Train:", train_x.shape, train_y.shape)
    print("Test:", test_x.shape, test_y.shape)

    # Currently in dataset, 0 is target and 1 is non-target - confusing.
    # Switch to normal convention (0=nontarget, 1=target)
    train_y = 1 - train_y.astype(int)
    test_y = 1 - test_y.astype(int)

    np.save(PROJECT_ROOT / "datasets" / "preprocessed" / f"train_x.seed_{seed}.npy", train_x)
    np.save(PROJECT_ROOT / "datasets" / "preprocessed" / f"train_y.seed_{seed}.npy", train_y)
    np.save(PROJECT_ROOT / "datasets" / "preprocessed" / f"test_x.seed_{seed}.npy", test_x)
    np.save(PROJECT_ROOT / "datasets" / "preprocessed" / f"test_y.seed_{seed}.npy", test_y)
    del train_x, train_y, test_x, test_y
