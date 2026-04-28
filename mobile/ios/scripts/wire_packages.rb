#!/usr/bin/env ruby
# Wire PoseKit + MediaPipeKit local Swift Packages into FollowThrough.xcodeproj,
# add the FollowThrough target's force-load + framework-search-path settings
# needed to consume MediaPipe at link time.
#
# Run ONCE during initial setup (or after a destructive Xcode pbxproj
# regeneration). The pbxproj is committed afterward and is the source of
# truth -- this script is just a reproducible setup ramp for new machines.
# Idempotent: safe to re-run.
#
# Usage:
#   cd mobile/ios && ruby scripts/wire_packages.rb

require 'xcodeproj'
require 'pathname'

repo_root = Pathname.new(__FILE__).realpath.dirname.parent.parent.parent
ios_root = repo_root.join('mobile/ios')
project_path = ios_root.join('FollowThrough/FollowThrough.xcodeproj')
project = Xcodeproj::Project.open(project_path.to_s)

app_target = project.targets.find { |t| t.name == 'FollowThrough' }
abort "FollowThrough target not found" unless app_target

# --------------------------------------------------------------------------
# 1. Add local Swift Package references for PoseKit + MediaPipeKit.
# --------------------------------------------------------------------------

PACKAGE_DEFS = [
  { name: 'PoseKit',      product: 'PoseKit',      relative: '../Packages/PoseKit' },
  { name: 'MediaPipeKit', product: 'MediaPipeKit', relative: '../Packages/MediaPipeKit' }
].freeze

PACKAGE_DEFS.each do |pkg|
  existing = project.root_object.package_references.find do |ref|
    ref.is_a?(Xcodeproj::Project::Object::XCLocalSwiftPackageReference) &&
      ref.relative_path == pkg[:relative]
  end

  if existing
    puts "  [skip] local package already referenced: #{pkg[:name]} -> #{pkg[:relative]}"
    next
  end

  ref = project.new(Xcodeproj::Project::Object::XCLocalSwiftPackageReference)
  ref.relative_path = pkg[:relative]
  project.root_object.package_references << ref
  puts "  [add]  local package reference: #{pkg[:name]} -> #{pkg[:relative]}"
end

# --------------------------------------------------------------------------
# 2. Add product references on the FollowThrough target so it actually
#    links the package products.
# --------------------------------------------------------------------------

PACKAGE_DEFS.each do |pkg|
  package_ref = project.root_object.package_references.find do |ref|
    ref.is_a?(Xcodeproj::Project::Object::XCLocalSwiftPackageReference) &&
      ref.relative_path == pkg[:relative]
  end

  product_dep = app_target.package_product_dependencies.find do |dep|
    dep.product_name == pkg[:product]
  end

  if product_dep
    puts "  [skip] target already links product: #{pkg[:product]}"
    next
  end

  dep = project.new(Xcodeproj::Project::Object::XCSwiftPackageProductDependency)
  dep.package = package_ref
  dep.product_name = pkg[:product]
  app_target.package_product_dependencies << dep

  # Add to Frameworks build phase so linking actually happens.
  fw_phase = app_target.frameworks_build_phase
  build_file = project.new(Xcodeproj::Project::Object::PBXBuildFile)
  build_file.product_ref = dep
  fw_phase.files << build_file

  puts "  [add]  target links product: #{pkg[:product]}"
end

# --------------------------------------------------------------------------
# 3. Force-load the graph_libraries .a files (sdk-conditional) and add
#    Vendor/MediaPipe to FRAMEWORK_SEARCH_PATHS so the binaryTarget
#    .frameworks resolve at app-link time.
# --------------------------------------------------------------------------

VENDOR_REL = '$(SRCROOT)/../Vendor/MediaPipe'.freeze

FORCE_LOAD_DEVICE = "-force_load \"#{VENDOR_REL}/graph_libraries/libMediaPipeTasksCommon_device_graph.a\"".freeze
FORCE_LOAD_SIM    = "-force_load \"#{VENDOR_REL}/graph_libraries/libMediaPipeTasksCommon_simulator_graph.a\"".freeze

def add_unique(setting, value)
  current = setting.is_a?(Array) ? setting.dup : (setting.nil? ? ['$(inherited)'] : [setting])
  return current if current.include?(value)
  current << value
  current
end

app_target.build_configurations.each do |config|
  bs = config.build_settings

  # FRAMEWORK_SEARCH_PATHS -- so MediaPipeTasksVision.framework is findable
  # when the app links MediaPipeKit.
  bs['FRAMEWORK_SEARCH_PATHS'] = add_unique(bs['FRAMEWORK_SEARCH_PATHS'], VENDOR_REL)

  # OTHER_LDFLAGS -- ObjC + c++ + force-load the right graph .a per SDK.
  base = add_unique(bs['OTHER_LDFLAGS'], '-ObjC')
  base = add_unique(base, '-lc++')
  bs['OTHER_LDFLAGS'] = base

  # SDK-conditional force_load.
  device_key = 'OTHER_LDFLAGS[sdk=iphoneos*]'
  sim_key    = 'OTHER_LDFLAGS[sdk=iphonesimulator*]'
  bs[device_key] = add_unique(bs[device_key], FORCE_LOAD_DEVICE)
  bs[sim_key]    = add_unique(bs[sim_key], FORCE_LOAD_SIM)

  puts "  [set]  config #{config.name}: framework search + force_load configured"
end

project.save
puts "Saved #{project_path}"
