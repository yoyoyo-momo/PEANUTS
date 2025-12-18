
# -*- coding: utf-8 -*-
"""
Class-based implementation for polling VIVOTEK thermal imaging parameters.
"""

import time
import datetime
import requests
from requests.auth import HTTPDigestAuth
from typing import Dict, Optional


class VivotekThermalPoller:
    def __init__(self, ip: str, username: str, password: str,
                 poll_interval: float = 2.0, timeout: int = 10, print_full: bool = False):
        self.ip = ip
        self.username = username
        self.password = password
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.print_full = print_full

        self.maxTemperature = 0.0

        self.cache: Dict[str, str] = {}
        self.last_raw: Optional[str] = None

    def fetch_raw(self) -> str:
        """Fetch raw text from the thermalImagerConfigureParam CGI."""
        url = f"http://{self.ip}/cgi-bin/param.cgi?action=get&type=areaTemperature&cameraID=1&areaID=0"
        resp = requests.get(url, auth=HTTPDigestAuth(self.username, self.password), timeout=self.timeout)
        resp.raise_for_status()
        return resp.text

    @staticmethod
    def parse_key_values(raw_text: str) -> Dict[str, str]:
        """Parse lines of 'key=value' into a dict. Normalize dotted keys to underscores."""
        kv = {}
        for line in raw_text.splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            kv[k.strip().replace(".", "_")] = v.strip()
        return kv

    def print_changes(self, changes: Dict[str, str], when: str) -> None:
        """Pretty-print only the changed keys/values."""
        if not changes:
            print(f"[{when}] No changes.")
            return
        print(f"[{when}] Updated keys ({len(changes)}):")
        for k, v in changes.items():
            print(f"  {k} = {v}")

    def poll_once(self) -> None:
        """Perform one poll cycle: fetch, parse, compare, and print changes."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            raw = self.fetch_raw()
            data = self.parse_key_values(raw)

            # Detect changes
            changes = {}
            for k, v in data.items():
                if k not in self.cache or self.cache[k] != v:
                    self.cache[k] = v
                    changes[k] = v
                    if k == "maxTemperature":
                        try:
                            self.maxTemperature = float(v)
                        except ValueError:
                            self.maxTemperature = 0.0
                        print(f"cached maxTemperature updated to: {self.maxTemperature}")

            # Detect removed keys
            removed = [k for k in list(self.cache.keys()) if k not in data]
            for k in removed:
                del self.cache[k]
            if removed:
                print(f"[{now}] Keys removed ({len(removed)}): {', '.join(removed)}")

            # Print results
            if self.print_full:
                print(f"[{now}] Full cache ({len(self.cache)} keys):")
                for k in sorted(self.cache.keys()):
                    print(f"  {k} = {self.cache[k]}")
            else:
                self.print_changes(changes, now)

            self.last_raw = raw

        except requests.HTTPError as e:
            print(f"[{now}] HTTP error: {e}")
        except requests.RequestException as e:
            print(f"[{now}] Request error: {e}")
        except Exception as e:
            print(f"[{now}] Unexpected error: {e}")

    def start_polling(self) -> None:
        """Start continuous polling until interrupted."""
        print("Starting VIVOTEK thermal parameter polling & caching...")
        while True:
            self.poll_once()
            time.sleep(self.poll_interval)


if __name__ == "__main__":
    poller = VivotekThermalPoller(
        ip="169.254.183.33",
        username="root",
        password="admin",
        poll_interval=2.0,
        timeout=10,
        print_full=False
    )
    poller.start_polling()
