"""Tests for scheduler config generation."""

from atomics.scheduler.cron import (
    generate_crontab_entry,
    generate_launchd_plist,
    generate_systemd_timer,
)


def test_crontab_entry_default():
    entry = generate_crontab_entry()
    assert "*/30" in entry
    assert "--tier baseline" in entry
    assert "--max-iterations 10" in entry
    assert "atomics-cron.log" in entry


def test_crontab_entry_custom_tier():
    entry = generate_crontab_entry(interval_minutes=15, max_iterations=5, tier="mega")
    assert "*/15" in entry
    assert "--tier mega" in entry
    assert "--max-iterations 5" in entry


def test_crontab_entry_custom_python():
    entry = generate_crontab_entry(python_path="/usr/bin/python3.11")
    assert "/usr/bin/python3.11" in entry


def test_systemd_timer_default():
    service, timer = generate_systemd_timer()
    assert "[Service]" in service
    assert "Type=oneshot" in service
    assert "--tier baseline" in service
    assert "[Timer]" in timer
    assert "OnCalendar=*:0/30" in timer
    assert "WantedBy=timers.target" in timer


def test_systemd_timer_custom():
    service, timer = generate_systemd_timer(
        interval_minutes=60, max_iterations=20, tier="ez", working_dir="/opt/atomics"
    )
    assert "--tier ez" in service
    assert "--max-iterations 20" in service
    assert "WorkingDirectory=/opt/atomics" in service
    assert "OnCalendar=*:0/60" in timer


def test_launchd_plist_default():
    plist = generate_launchd_plist()
    assert "com.babywyrm.atomics.baseline" in plist
    assert "<integer>1800</integer>" in plist
    assert "--tier" in plist
    assert "baseline" in plist
    assert "RunAtLoad" in plist


def test_launchd_plist_custom():
    plist = generate_launchd_plist(
        interval_minutes=10, max_iterations=3, tier="mega", working_dir="/tmp/test"
    )
    assert "com.babywyrm.atomics.mega" in plist
    assert "<integer>600</integer>" in plist
    assert "--tier" in plist
    assert "mega" in plist
    assert "/tmp/test" in plist


def test_all_formats_include_tier():
    for tier in ("ez", "baseline", "mega"):
        entry = generate_crontab_entry(tier=tier)
        assert f"--tier {tier}" in entry

        service, _ = generate_systemd_timer(tier=tier)
        assert f"--tier {tier}" in service

        plist = generate_launchd_plist(tier=tier)
        assert f"<string>{tier}</string>" in plist
