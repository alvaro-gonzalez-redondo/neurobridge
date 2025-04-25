import torch
from neurobridge.bridges import AxonalBridge


def test_axonal_bridge_local():
    size = 16  # Número de axones
    n_steps = 4  # Periodo de comunicación
    device = "cpu"

    bridge = AxonalBridge(
        size=size, n_steps=n_steps, rank=0, world_size=1, device=device
    )

    # Paso 0: escribir spikes en las neuronas 3 y 5
    spikes = torch.zeros(size, dtype=torch.uint8)
    spikes[3] = 1
    spikes[5] = 1
    bridge.write_spikes(spikes)
    bridge.step()  # Avanzamos un paso

    # Paso 1-3: spikes irrelevantes
    for _ in range(n_steps - 1):
        bridge.write_spikes(torch.zeros(size, dtype=torch.uint8))
        bridge.step()

    # Paso 4: debería volver a leer lo escrito en el paso 0
    read_spikes = bridge.read_spikes()
    assert read_spikes[3] == 1, "Spike en neurona 3 no recibido correctamente"
    assert read_spikes[5] == 1, "Spike en neurona 5 no recibido correctamente"
    assert read_spikes.sum() == 2, "Número de spikes recibidos incorrecto"

    print("✅ AxonalBridge funciona correctamente en modo local (no distribuido).")


if __name__ == "__main__":
    test_axonal_bridge_local()
