"""FLUX Bytecode VM — Clean Python Implementation"""
class FluxVM:
    def __init__(self, bytecode):
        self.gp = [0]*16; self.pc = 0; self.halted = False; self.cycles = 0; self.stack = []; self.bc = bytecode
    def _u8(self):
        v = self.bc[self.pc]; self.pc += 1; return v
    def _i16(self):
        lo = self.bc[self.pc]; hi = self.bc[self.pc+1]; self.pc += 2
        return lo | (hi << 8) if hi < 128 else lo | (hi << 8) - 65536
    def execute(self):
        self.halted = False; self.cycles = 0
        while not self.halted and self.pc < len(self.bc) and self.cycles < 10000000:
            op = self._u8(); self.cycles += 1
            if op == 0x80: self.halted = True
            elif op == 0x01: d,s = self._u8(),self._u8(); self.gp[d]=self.gp[s]
            elif op == 0x2B: d = self._u8(); v = self._i16(); self.gp[d]=v
            elif op == 0x08: d,a,b = self._u8(),self._u8(),self._u8(); self.gp[d]=self.gp[a]+self.gp[b]
            elif op == 0x09: d,a,b = self._u8(),self._u8(),self._u8(); self.gp[d]=self.gp[a]-self.gp[b]
            elif op == 0x0A: d,a,b = self._u8(),self._u8(),self._u8(); self.gp[d]=self.gp[a]*self.gp[b]
            elif op == 0x0B: d,a,b = self._u8(),self._u8(),self._u8(); self.gp[d]=int(self.gp[a]/self.gp[b])
            elif op == 0x0E: self.gp[self._u8()] += 1
            elif op == 0x0F: self.gp[self._u8()] -= 1
            elif op == 0x06:
                d = self._u8(); off = self._i16()
                if self.gp[d] != 0: self.pc += off
            elif op == 0x2E:
                d = self._u8(); off = self._i16()
                if self.gp[d] == 0: self.pc += off
            elif op == 0x07: self.pc += self._i16()
            elif op == 0x10: self.stack.append(self.gp[self._u8()])
            elif op == 0x11: self.gp[self._u8()] = self.stack.pop()
            elif op == 0x2D: a,b = self._u8(),self._u8(); self.gp[13] = (self.gp[a]>self.gp[b])-(self.gp[a]<self.gp[b])
            else: raise ValueError(f"Unknown opcode: 0x{op:02X}")
        return self.cycles

class A2AAgent:
    def __init__(self, agent_id, bytecode, role="worker"):
        self.id=agent_id; self.vm=FluxVM(bytecode); self.role=role; self.trust=1.0; self.inbox=[]
    def step(self): self.vm.execute()
    def tell(self, other, payload): other.inbox.append({"from":self.id,"type":"TELL","payload":payload})

class Swarm:
    def __init__(self): self.agents = {}
    def add(self, agent): self.agents[agent.id] = agent
    def tick(self):
        for a in self.agents.values(): a.step()

if __name__ == "__main__":
    print("╔════════════════════════════════════════╗")
    print("║   FLUX.py — Clean Python VM           ║")
    print("║   SuperInstance / Oracle1              ║")
    print("╚════════════════════════════════════════╝\n")
    fact = bytes([0x2B,0x00,0x07,0x00, 0x2B,0x01,0x01,0x00, 0x0A,0x01,0x01,0x00, 0x0F,0x00, 0x06,0x00,0xF6,0xFF, 0x80])
    vm = FluxVM(fact); vm.execute()
    print(f"Factorial(7): R1 = {vm.gp[1]} (expect 5040)")
    import time
    I=10000; t0=time.monotonic()
    for _ in range(I): FluxVM(fact).execute()
    t=time.monotonic()-t0
    print(f"Benchmark (10K): {t*1000:.1f} ms | {t*1e9/I:.0f} ns/iter")
    s=Swarm()
    for i in range(5): s.add(A2AAgent(f"a{i}",fact,["worker","scout"][i%2]))
    s.tick()
    print(f"\nSwarm: {len(s.agents)} agents ticked ✓")
