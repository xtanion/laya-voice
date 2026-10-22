#include <Arduino.h>
#include <NimBLEDevice.h>
#include <driver/i2s.h>

#ifndef PIN_MIC_SCK
#define PIN_MIC_SCK 26
#endif
#ifndef PIN_MIC_WS
#define PIN_MIC_WS 25
#endif
#ifndef PIN_MIC_SD
#define PIN_MIC_SD 33
#endif
#ifndef PIN_BUTTON
#define PIN_BUTTON 0
#endif
#ifndef PIN_LED
#define PIN_LED 2
#endif
// 24-bit samples arrive left-aligned in 32 bits; >>16 is unity, each step lower doubles the gain.
// Print the mic level every 2 s while idle, to check wiring without the button.
#ifndef MIC_MONITOR
#define MIC_MONITOR 1
#endif
#ifndef GAIN_SHIFT
#define GAIN_SHIFT 13
#endif

static const char *DEVICE_NAME = "Laya Voice";
static const char *SERVICE_UUID = "3e1c1517-cff3-47bc-bcb3-b0276cdbfdfd";
static const char *AUDIO_UUID = "3e1c1518-cff3-47bc-bcb3-b0276cdbfdfd";

static const int SAMPLE_RATE = 16000;
static const int MAX_PACKET_SAMPLES = 476;
static const uint16_t MIN_MTU = 100;
static const uint32_t MAX_RECORD_MS = 14000;
static const uint32_t DEBOUNCE_MS = 25;
static const uint8_t PKT_START = 0x01, PKT_AUDIO = 0x02, PKT_STOP = 0x03;

static const int16_t STEPS[89] = {
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97,
    107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428,
    4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350,
    22385, 24623, 27086, 29794, 32767};
static const int8_t INDEX_SHIFT[8] = {-1, -1, -1, -1, 2, 4, 6, 8};

static NimBLECharacteristic *audioChr = nullptr;
static volatile bool linked = false;
static volatile bool secure = false;
static volatile bool subscribed = false;
static volatile uint16_t mtu = 23;

static bool recording = false;
static bool waitRelease = false;
static uint32_t recordStart = 0;
static uint32_t samplesSent = 0;
static uint16_t seq = 0;
static int16_t peak = 0;
static uint32_t packetsSent = 0;
static uint32_t notifyRetries = 0;
static int32_t adpcmPredictor = 0;
static int adpcmIndex = 0;
static int32_t raw[MAX_PACKET_SAMPLES];
static int16_t pcm[MAX_PACKET_SAMPLES];
static uint8_t packet[6 + MAX_PACKET_SAMPLES / 2];

class ServerCb : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer *s, NimBLEConnInfo &info) override {
        linked = true;
        s->setDataLen(info.getConnHandle(), 251);
        s->updateConnParams(info.getConnHandle(), 12, 12, 0, 400);
        NimBLEDevice::startSecurity(info.getConnHandle());
        Serial.printf("connected: %s, interval %.1f ms\n", info.getAddress().toString().c_str(), info.getConnInterval() * 1.25f);
    }
    void onConnParamsUpdate(NimBLEConnInfo &info) override {
        Serial.printf("conn params: interval %.1f ms, latency %u, timeout %u ms\n", info.getConnInterval() * 1.25f,
                      info.getConnLatency(), info.getConnTimeout() * 10);
    }
    void onDisconnect(NimBLEServer *s, NimBLEConnInfo &info, int reason) override {
        linked = false;
        secure = false;
        subscribed = false;
        mtu = 23;
        Serial.printf("disconnected (reason %d)\n", reason);
    }
    void onMTUChange(uint16_t m, NimBLEConnInfo &info) override {
        mtu = m;
        Serial.printf("mtu %u\n", m);
    }
    void onAuthenticationComplete(NimBLEConnInfo &info) override {
        secure = info.isEncrypted();
        Serial.println(secure ? "link encrypted" : "pairing failed, disconnecting");
        if (!secure) NimBLEDevice::getServer()->disconnect(info.getConnHandle());
    }
};

class AudioCb : public NimBLECharacteristicCallbacks {
    void onSubscribe(NimBLECharacteristic *c, NimBLEConnInfo &info, uint16_t subValue) override {
        subscribed = subValue & 0x01;
        Serial.printf("subscribed %d\n", (int)subscribed);
    }
};

static bool ready() { return linked && secure && subscribed && mtu >= MIN_MTU; }

static int packetSamples() {
    int n = (int)(mtu - 3 - 6) * 2;
    return n > MAX_PACKET_SAMPLES ? MAX_PACKET_SAMPLES : n;
}

static void micBegin(int sck, int ws, bool right) {
    i2s_config_t cfg = {};
    cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
    cfg.sample_rate = SAMPLE_RATE;
    cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
    cfg.channel_format = right ? I2S_CHANNEL_FMT_ONLY_RIGHT : I2S_CHANNEL_FMT_ONLY_LEFT;
    cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
    cfg.dma_desc_num = 16;
    cfg.dma_frame_num = 256;
    cfg.use_apll = false;

    i2s_pin_config_t pins = {};
    pins.mck_io_num = I2S_PIN_NO_CHANGE;
    pins.bck_io_num = sck;
    pins.ws_io_num = ws;
    pins.data_out_num = I2S_PIN_NO_CHANGE;
    pins.data_in_num = PIN_MIC_SD;

    i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pins);
}

static size_t micRead(int samples, TickType_t wait) {
    size_t bytes = 0;
    i2s_read(I2S_NUM_0, raw, samples * sizeof(int32_t), &bytes, wait);
    size_t n = bytes / sizeof(int32_t);
    for (size_t i = 0; i < n; i++) {
        int32_t s = raw[i] >> GAIN_SHIFT;
        if (s > 32767) s = 32767;
        if (s < -32768) s = -32768;
        pcm[i] = (int16_t)s;
        int16_t a = s < 0 ? -s : s;
        if (a > peak) peak = a;
    }
    return n;
}

// Tries SCK/WS both ways round and both channel slots, keeps the one that yields data,
// so a swapped clock pair or L/R strap does not leave the mic silent.
static void micProbe() {
    const int configs[4][3] = {
        {PIN_MIC_SCK, PIN_MIC_WS, 0}, {PIN_MIC_SCK, PIN_MIC_WS, 1}, {PIN_MIC_WS, PIN_MIC_SCK, 0}, {PIN_MIC_WS, PIN_MIC_SCK, 1}};
    int best = 0;
    uint32_t bestNonzero = 0;
    for (int c = 0; c < 4; c++) {
        micBegin(configs[c][0], configs[c][1], configs[c][2]);
        delay(150);
        uint32_t nonzero = 0, total = 0;
        int32_t maxAbs = 0;
        while (total < 8000) {
            size_t bytes = 0;
            i2s_read(I2S_NUM_0, raw, sizeof(raw), &bytes, pdMS_TO_TICKS(100));
            if (!bytes) break;
            for (size_t i = 0; i < bytes / sizeof(int32_t); i++) {
                int32_t v = raw[i] >> 8;
                if (v) nonzero++;
                if (abs(v) > maxAbs) maxAbs = abs(v);
            }
            total += bytes / sizeof(int32_t);
        }
        Serial.printf("mic probe sck=%d ws=%d %s: %lu/%lu nonzero, max %ld\n", configs[c][0], configs[c][1],
                      configs[c][2] ? "right" : "left", (unsigned long)nonzero, (unsigned long)total, (long)maxAbs);
        if (nonzero > bestNonzero) {
            bestNonzero = nonzero;
            best = c;
        }
        i2s_driver_uninstall(I2S_NUM_0);
    }
    micBegin(configs[best][0], configs[best][1], configs[best][2]);
    if (bestNonzero == 0) Serial.println("mic probe: no data on any setup, check SD/VDD/GND wiring");
    else Serial.printf("mic probe: using sck=%d ws=%d %s\n", configs[best][0], configs[best][1], configs[best][2] ? "right" : "left");
}

static void micFlush() {
    while (micRead(MAX_PACKET_SAMPLES, 0) > 0) {
    }
}

static uint8_t adpcmEncode(int16_t sample) {
    int step = STEPS[adpcmIndex];
    int diff = sample - adpcmPredictor;
    uint8_t code = 0;
    if (diff < 0) {
        code = 8;
        diff = -diff;
    }
    int delta = step >> 3;
    if (diff >= step) {
        code |= 4;
        diff -= step;
        delta += step;
    }
    if (diff >= (step >> 1)) {
        code |= 2;
        diff -= step >> 1;
        delta += step >> 1;
    }
    if (diff >= (step >> 2)) {
        code |= 1;
        delta += step >> 2;
    }
    adpcmPredictor += (code & 8) ? -delta : delta;
    if (adpcmPredictor > 32767) adpcmPredictor = 32767;
    if (adpcmPredictor < -32768) adpcmPredictor = -32768;
    adpcmIndex += INDEX_SHIFT[code & 7];
    if (adpcmIndex < 0) adpcmIndex = 0;
    if (adpcmIndex > 88) adpcmIndex = 88;
    return code;
}

static bool notify(const uint8_t *buf, size_t len) {
    for (int i = 0; i < 100; i++) {
        if (!ready()) return false;
        if (audioChr->notify(buf, len)) {
            packetsSent++;
            return true;
        }
        notifyRetries++;
        delay(5);
    }
    return false;
}

static bool sendControl(uint8_t type) {
    uint8_t buf[3] = {type, (uint8_t)(seq & 0xFF), (uint8_t)(seq >> 8)};
    return notify(buf, sizeof(buf));
}

static bool sendAudio(size_t n) {
    seq++;
    packet[0] = PKT_AUDIO;
    packet[1] = seq & 0xFF;
    packet[2] = seq >> 8;
    packet[3] = adpcmPredictor & 0xFF;
    packet[4] = (adpcmPredictor >> 8) & 0xFF;
    packet[5] = adpcmIndex;
    for (size_t i = 0; i < n; i += 2) {
        uint8_t lo = adpcmEncode(pcm[i]);
        uint8_t hi = i + 1 < n ? adpcmEncode(pcm[i + 1]) : 0;
        packet[6 + i / 2] = lo | (hi << 4);
    }
    return notify(packet, 6 + (n + 1) / 2);
}

static bool buttonDown() {
    static bool stable = false;
    static bool last = false;
    static uint32_t changed = 0;
    bool now = digitalRead(PIN_BUTTON) == LOW;
    if (now != last) {
        last = now;
        changed = millis();
    }
    if (millis() - changed >= DEBOUNCE_MS) stable = last;
    return stable;
}

static void finish(const char *why) {
    recording = false;
    Serial.printf("%s: %lu ms audio in %lu ms, %lu packets, %lu retries, peak %d\n", why,
                  (unsigned long)(samplesSent * 1000UL / SAMPLE_RATE), (unsigned long)(millis() - recordStart),
                  (unsigned long)packetsSent, (unsigned long)notifyRetries, peak);
    if (peak == 0) Serial.println("peak is 0: check mic wiring or flip MIC_RIGHT_CHANNEL");
}

static void updateLed() {
    uint32_t t = millis();
    bool on;
    if (recording) on = true;
    else if (!linked) on = (t / 600) % 2;
    else if (!ready()) on = (t / 150) % 2;
    else on = (t % 3000) < 60;
    digitalWrite(PIN_LED, on ? HIGH : LOW);
}

void setup() {
    setCpuFrequencyMhz(80);
    Serial.begin(115200);
    pinMode(PIN_BUTTON, INPUT_PULLUP);
    pinMode(PIN_LED, OUTPUT);
    delay(500);
    micProbe();

    NimBLEDevice::init(DEVICE_NAME);
    NimBLEDevice::setMTU(247);
    NimBLEDevice::setSecurityAuth(true, false, true);
    NimBLEDevice::setSecurityIOCap(BLE_HS_IO_NO_INPUT_OUTPUT);

    NimBLEServer *server = NimBLEDevice::createServer();
    server->setCallbacks(new ServerCb());
    server->advertiseOnDisconnect(true);
    NimBLEService *svc = server->createService(SERVICE_UUID);
    audioChr = svc->createCharacteristic(AUDIO_UUID, NIMBLE_PROPERTY::NOTIFY);
    audioChr->setCallbacks(new AudioCb());
    server->start();

    NimBLEAdvertisementData advData;
    advData.setFlags(BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP);
    advData.addServiceUUID(NimBLEUUID(SERVICE_UUID));
    NimBLEAdvertisementData scanData;
    scanData.setName(DEVICE_NAME);
    NimBLEAdvertising *adv = NimBLEDevice::getAdvertising();
    adv->setAdvertisementData(advData);
    adv->setScanResponseData(scanData);
    adv->start();
    Serial.println("advertising as Laya Voice");
}

void loop() {
    updateLed();

    bool down = buttonDown();
    static bool wasDown = false;
    if (down != wasDown) {
        wasDown = down;
        Serial.printf("button %s%s\n", down ? "down" : "up",
                      down && !ready() ? (linked ? " (not ready: pairing/subscribe/mtu)" : " (not connected)") : "");
    }
    if (!down) waitRelease = false;

    if (!recording) {
        if (down && !waitRelease && ready()) {
            micFlush();
            peak = 0;
            samplesSent = 0;
            packetsSent = 0;
            notifyRetries = 0;
            seq = 0;
            adpcmPredictor = 0;
            adpcmIndex = 0;
            recordStart = millis();
            if (sendControl(PKT_START)) {
                recording = true;
            } else {
                waitRelease = true;
            }
        } else {
            micRead(MAX_PACKET_SAMPLES, pdMS_TO_TICKS(5));
#if MIC_MONITOR
            static uint32_t lastLevel = 0;
            if (millis() - lastLevel >= 2000) {
                lastLevel = millis();
                Serial.printf("mic level: peak %d\n", peak);
                peak = 0;
            }
#endif
        }
        return;
    }

    size_t n = micRead(packetSamples(), pdMS_TO_TICKS(50));
    if (n > 0) {
        if (!sendAudio(n)) {
            waitRelease = true;
            finish("link lost mid-recording");
            return;
        }
        samplesSent += n;
    }
    if (!down) {
        seq++;
        sendControl(PKT_STOP);
        finish("sent");
    } else if (millis() - recordStart >= MAX_RECORD_MS) {
        seq++;
        sendControl(PKT_STOP);
        waitRelease = true;
        finish("max length reached");
    }
}
