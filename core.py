#Energy Consumption Comparison of Interactive Cloud-Based and Local Applications
class Core_Entity:
    def __init__(self,id):
        self.id=id
        self.speed_UL=None
        self.speed_DL=None
        self.energybit_UL=None
        self.energybit_DL=None

class Core:
    def __init__(self,hops):
        self.elements=[]

        self.energy_UL=  0
        self.energy_DL = 0
        self.time_UL = 0
        self.time_DL = 0

        bs=Core_Entity('bs')
        bs.speed_DL=72 # Mbps
        bs.speed_UL=72 # Mbps
        bs.energybit_UL=19e-6 # joule per bit
        bs.energybit_DL=76.2e-6
        self.elements.append(bs)

        ethswitch_metro=Core_Entity('ethswitch_metro')
        ethswitch_metro.speed_DL=256000 # Mbps
        ethswitch_metro.speed_UL=256000 # Mbps
        ethswitch_metro.energybit_UL=21.4e-9 # joule per bit
        ethswitch_metro.energybit_DL=21.4e-9
        self.elements.append(ethswitch_metro)

        bng=Core_Entity('bng')
        bng.speed_DL=320000 # Mbps
        bng.speed_UL=320000 # Mbps
        bng.energybit_UL=18.3e-9 # joule per bit
        bng.energybit_DL=18.3e-9
        self.elements.append(bng)

        edgerouter1=Core_Entity('edgerouter1')
        edgerouter1.speed_DL=560000 # Mbps
        edgerouter1.speed_UL=560000 # Mbps
        edgerouter1.energybit_UL=25.2e-9 # joule per bit
        edgerouter1.energybit_DL=25.2e-9
        self.elements.append(edgerouter1)

        for hop in range(0,hops):
            corerouter=Core_Entity('corerouter'+str(hop+1))
            corerouter.speed_DL = 4480000  # Mbps
            corerouter.speed_UL = 4480000  # Mbps
            corerouter.energybit_UL = 8.5e-9  # joule per bit
            corerouter.energybit_DL = 8.5e-9
            self.elements.append(corerouter)

        edgerouter2=Core_Entity('edgerouter2')
        edgerouter2.speed_DL=560000 # Mbps
        edgerouter2.speed_UL=560000 # Mbps
        edgerouter2.energybit_UL=25.2e-9 # joule per bit
        edgerouter2.energybit_DL=25.2e-9
        self.elements.append(edgerouter2)

        dcswitch=Core_Entity('dcswitch')
        dcswitch.speed_DL=320000 # Mbps
        dcswitch.speed_UL=320000 # Mbps
        dcswitch.energybit_UL=19.6e-9 # joule per bit
        dcswitch.energybit_DL=19.6e-9
        self.elements.append(dcswitch)

    def print_chain(self):
        for elm in self.elements:
            print(elm.id)

    def calc_time_UL(self,mb):
        total_time=0
        for elm in self.elements:
            if elm.id=='bs':
                time_per_elm=0
            else:
                time_per_elm=mb/elm.speed_UL
            total_time=total_time+time_per_elm
        return total_time

    def calc_time_DL(self,mb):
        total_time=0
        for elm in self.elements:
            if elm.id=='bs':
                time_per_elm=0
            else:
                time_per_elm=mb/elm.speed_DL
            total_time=total_time+time_per_elm
        return total_time

    def add_energy_UL(self,mb):
        for elm in self.elements:
            self.energy_UL=self.energy_UL+elm.energybit_UL * mb* 8e6

    def add_energy_DL(self,mb):
        for elm in self.elements:
            self.energy_DL=self.energy_DL+elm.energybit_DL * mb* 8e6

    def add_time_UL(self,mb):
        self.time_UL = self.time_UL + self.calc_time_UL(mb)

    def add_time_DL(self,mb):
        self.time_DL = self.time_DL + self.calc_time_DL(mb)