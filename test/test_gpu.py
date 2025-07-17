import torch as t


def test_gpu():
    print(t.cuda.get_device_name())

if __name__ == "__main__":
    test_gpu()
    print("Test passed successfully.")