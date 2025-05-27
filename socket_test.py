from bt import BT
import time

class main():
    def __init__(self) -> None:
        self.bt_dev = []
        self.bt = BT()
        self.devices_mac_select()
        self.device_connect()
        
    def devices_mac_select(self):
        print('Search for Bluetooth devices.\n')
        r = self.bt.find_devices()
        
        if len(r) != 0:
            for i in range(0, len(r)):
                name = r[i]['name'].ljust(30)
                addr = r[i]['addr']
                num = str(i).ljust(4)

                self.bt_dev.append([r[i]['name'], r[i]['addr']])

                print(f'NUM : {num} // DEVICE NAME : {name} // MAC ADDRESS : {addr}')
                
            print('')
        else:
            print('There are no paired devices.')
            print('Try again after Bluetooth pairing\n')
            _sys.exit()
    def device_connect(self):
        print('Enter the number of devices to connect')

        while True:
            try:
                id = int(input('NUM : '))
                target_data = self.bt_dev[id]
            except ValueError:
                print('An invalid value was entered.')
            except IndexError:
                print('There is no device with that number.')
            else:
                break

        print(f'\nCONNECT -> {target_data[0]} : {target_data[1]}\n')

        self.sock = self.bt.connect(target_data[1])

        if self.sock != None:
            print('The device is connected normally.\n')
        else:
            print('Bluetooth connection failed.')
            print("Check the device's power and try again...\n")
            _sys.exit()
        
        cnt = 0
        while True:
            message = "Hello{}".format(cnt)
            self.sock.send(message)
            print("send: ", message)
            
            recv_message = self.sock.recv(1024)
            if recv_message:
                print("Received: ", recv_message.decode())
            else:
                self.sock.close()
                break
            cnt += 1
            time.sleep(0.1)


if __name__ == '__main__':
    app = main()
