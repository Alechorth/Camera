import serial
import time
import math

# Commandes à envoyer à la camera

SYNC = [0xAA, 0x0D, 0x00, 0x00, 0x00, 0x00]
ACK = [0xAA, 0x0e, 0x0D, 0x00, 0x00, 0x00]
INIT = [0xAA, 0x01, 0x00, 0x07, 0x07, 0x07]
PKG_SIZE = [0xAA, 0x06, 0x08, 0x00, 0x02, 0x02]
SNAPSHOT = [0xAA, 0x05, 0x00, 0x00, 0x00, 0x00]
PICTURE = [0xAA, 0x04, 0x01, 0x00, 0x00, 0x00]
RESET = [0xAA, 0x08, 0x00, 0x00, 0x00, 0xFF]


class UCam(serial.Serial):
    """Classe définissant la caméra """

    def __init__(self, port="COM4"):
        """Initialise la classe
        ATTENTION le port est à verifier dans device manager """
        serial.Serial.__init__(self, port, baudrate=921600, timeout=0.01)
        print("Connected")
        self.synced = False

    def sync(self):
        """ Syncronise la caméra avec l'ordi """
        numTries = 60
        while numTries > 0 and self.synced == False:
            self.write(bytearray(SYNC))
            read = self.read(6)
            print(read[:3],)
            if read[:3] == bytearray([0xAA, 0x0e, 0x0D]) and len(read) == 6:
                if self.read(6) == bytearray(SYNC):
                    self.write(bytearray(ACK))
                    self.synced = True
                    print("SYNC")
            numTries -= 1
            time.sleep(0.05)

    def takePicture(self, name):
        """Prend une photo, nom de la photo en paramètre """
        self.setPictureSize()
        self.setPkgSize()
        self.snapshot()
        self.getPicture(name)

    def setPictureSize(self):
        """Règle la taille de la photo """
        self.write(bytearray(INIT))
        assert self.waitByte(6)[:3] == bytearray(
            [0xAA, 0x0e, 0x01]), "Mauvais retour"

    def setPkgSize(self):
        """Règle la taille des packets pour le transfert """
        self.write(bytearray(PKG_SIZE))
        assert self.waitByte(6)[:3] == bytearray(
            [0xAA, 0x0e, 0x06]), "Mauvais retour"

    def snapshot(self):
        """Prend la photo """
        self.write(bytearray(SNAPSHOT))
        assert self.waitByte(6)[:3] == bytearray(
            [0xAA, 0x0e, 0x05]), "Mauvais retour"

    def getPicture(self, name):
        """Transfert la photo sur l'ordi"""
        t = time.time()
        self.write(bytearray(PICTURE))
        assert self.waitByte(6)[:3] == bytearray(
            [0xAA, 0x0e, 0x04]), "Mauvais retour"

        data = self.waitByte(6)

        assert data[:3] == bytearray([0xAA, 0x0a, 0x01]), "Mauvais retour"
        imgSize = int.from_bytes(data[3:], byteorder="little")
        print("Taille de limage ", imgSize)

        self.write(bytearray([0xAA, 0x0e, 0x00, 0x00, 0x00, 0x00]))

        nbPackage = math.floor(imgSize / (512 - 6))

        with open(name, "w+b") as f:
            for i in range(1, nbPackage + 1):
                read = self.waitByte(512)
                f.write(read[4:-2])

                self.write(bytearray([0xAA, 0x0E, 0x00, 0x00, i, 0x00]))

            f.write(self.waitByte(imgSize - nbPackage * (512 - 6) + 2)[4:-2])

        print("Picture Taken in ", time.time() - t, "s")

    def waitByte(self, nb):
        """Fonction d'attente des données"""
        array = bytearray(nb)
        cur = 0
        while cur < nb:
            read = self.read(1)
            if len(read) == 1:
                array[cur] = read[0]
                cur += 1
        return array

    def reset(self):
        """Reset la caméra"""
        self.write(bytearray(RESET))


def main():
    """Code principal du programme"""

    cam = UCam()
    # Syncronise la caméra
    cam.sync()

    print(cam.synced)

    # Si snycro, prend 5 photo toutes les 0.5s (plus temps de transfert)
    if cam.synced:
        for i in range(5):
            cam.takePicture("{}.jpg".format(i))
            time.sleep(0.5)
            cam.reset()
            cam.synced = False
            cam.sync()


if __name__ == "__main__":
    main()
